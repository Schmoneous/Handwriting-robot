
"""
limit_switch_homing.py
 
Limit-switch homing/calibration module for DrawCore (EBB firmware v2.6.5,
CoreXY kinematics). Designed to plug into the existing EBB serial pipeline.
 
Hardware assumption (update if different):
    - One limit switch wired to Port C, Pin 3 (PC3) on the unused IO header.
    - Switch reads LOW (0) at rest, HIGH (1) when pressed.
      (If your wiring is inverted, flip TRIGGERED_VALUE below.)
 
Firmware note:
    EBB v2.6.5 does NOT have the automatic CU,51/52/53 limit-switch-stop
    feature (that was added in firmware v3.0). So this module actively
    polls the pin in software and issues an ES (E-Stop) the instant a
    press is confirmed, rather than relying on the board to stop itself.
 
Extending to two switches (full X/Y homing):
    Duplicate the LimitSwitch instance for a second port/pin, and run
    home_axis() once per physical axis before combining into your
    CoreXY move sequence. See the __main__ block at the bottom for
    an example skeleton.
"""
 
import serial
import time
import threading
 
 
class EBBLimitSwitchHomer:
    def __init__(self, port_path, baud=115200, switch_port='C', switch_pin='3',
                 triggered_value='1', steps_per_mm=80, flip_x=True):
        """
        port_path: e.g. '/dev/tty.usbmodem11401'
        switch_port / switch_pin: EBB port letter + pin number the switch is wired to
        triggered_value: the PI response value ('0' or '1') that means "pressed"
        steps_per_mm / flip_x: carried over from your calibrated hardware constants
        """
        self.ser = serial.Serial(port_path, baud, timeout=0.2)
        time.sleep(2)  # let the EBB's USB-CDC settle after connect
        self.switch_port = switch_port
        self.switch_pin = switch_pin
        self.triggered_value = triggered_value
        self.steps_per_mm = steps_per_mm
        self.flip_x = flip_x
        self._lock = threading.Lock()
 
    # ---------- low-level EBB command helpers ----------
 
    def _send(self, cmd):
        with self._lock:
            self.ser.reset_input_buffer()
            self.ser.write((cmd + '\r').encode())
            return self.ser.readline().decode(errors='replace').strip()
 
    def configure_switch_pin(self):
        """Set the limit switch pin as a digital input (PD,<port>,<pin>,1)."""
        resp = self._send(f'PD,{self.switch_port},{self.switch_pin},1')
        if resp != 'OK':
            raise RuntimeError(f'Failed to configure switch pin as input: {resp!r}')
 
    def read_switch_raw(self):
        """Single PI read. Returns '0' or '1' as a string, or None on bad response."""
        resp = self._send(f'PI,{self.switch_port},{self.switch_pin}')
        # EBB v2.6.5 legacy response format is "PI,<value>"
        if resp.startswith('PI,'):
            return resp.split(',')[1]
        return None
 
    def is_triggered_debounced(self, confirm_count=3, poll_interval_s=0.005):
        """
        Polls the switch confirm_count times in a row, poll_interval_s apart.
        Returns True only if every read matches triggered_value (debounced press).
        Returns False immediately on the first non-matching read (fast rejection
        of bounce/noise so this can be called tightly in a homing loop).
        """
        for _ in range(confirm_count):
            val = self.read_switch_raw()
            if val != self.triggered_value:
                return False
            time.sleep(poll_interval_s)
        return True
 
    def e_stop(self, disable_motors=False):
        """Immediately halt any in-progress SM move and flush the FIFO."""
        cmd = 'ES,1' if disable_motors else 'ES'
        return self._send(cmd)
 
    def enable_motors(self, mode=1):
        """Re-enable both motors after an E-Stop (also resets the ES-triggered halt)."""
        return self._send(f'EM,{mode},{mode}')
 
    # ---------- homing routine ----------
 
    def home_axis(self, step_chunk=20, chunk_duration_ms=40,
                   direction=-1, max_travel_mm=300, confirm_count=3):
        """
        Moves the CoreXY system in small steps toward the limit switch until
        it triggers (debounced), then stops immediately and zeroes position.
 
        step_chunk: motor steps per SM chunk (small = finer stop resolution,
                    but more serial traffic; 20 steps at steps_per_mm=80 is
                    ~0.25mm resolution)
        direction: -1 or 1, sets travel direction. Adjust to point the
                   carriage toward your switch.
        max_travel_mm: safety cutoff if the switch never triggers (prevents
                       ramming the frame if wiring/logic is wrong)
 
        NOTE: This drives axis1/axis2 equally, which for CoreXY moves the
        carriage along one diagonal, not a pure physical X or Y. Replace the
        SM,<duration>,<axis1>,<axis2> line below with your actual CoreXY
        transform (the same one your XM/SM wrapper already uses for
        physical-to-motor step conversion) so it moves along the correct
        physical axis for the switch you're homing against.
        """
        self.configure_switch_pin()
 
        max_chunks = int((max_travel_mm * self.steps_per_mm) / step_chunk)
        signed_step = direction * step_chunk
 
        for _ in range(max_chunks):
            if self.is_triggered_debounced(confirm_count=confirm_count):
                self.e_stop()
                self._send('CS')  # zero the global step position at the switch
                return True
 
            # TODO: replace with your real CoreXY axis1/axis2 step mapping
            self._send(f'SM,{chunk_duration_ms},{signed_step},{signed_step}')
 
            # Give the chunk time to execute before polling/issuing the next one
            time.sleep(chunk_duration_ms / 1000.0)
 
        # Safety cutoff hit without a trigger — stop and report failure
        self.e_stop()
        return False
 
    def close(self):
        self.ser.close()
 
 
if __name__ == '__main__':
    # Example standalone test — adjust port path to match your machine
    homer = EBBLimitSwitchHomer('/dev/tty.usbmodem11401', switch_port='C', switch_pin='3')
    try:
        print('Homing... (move carriage toward switch by hand if testing without motors)')
        success = homer.home_axis()
        print('Homed successfully.' if success else 'Homing FAILED — switch never triggered.')
    finally:
        homer.close()
 