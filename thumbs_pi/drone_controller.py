import time
import logging
import threading
import warnings
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=DeprecationWarning)

CF_URI = "radio://0/10/2M"

# Tuned constants
HOVER_THRUST = 37000   # your magic value
HOVER_TIME   = 1.8     # seconds in the air
DESCENT_FACTOR = 0.75  # fraction of hover for controlled descent

# Avoid overlapping flights
_flight_lock = threading.Lock()
_is_flying = False


def _send_thrust(cf: Crazyflie, thrust: int, duration_s: float):
    """Send constant thrust (no roll/pitch) for duration_s."""
    start = time.time()
    while time.time() - start < duration_s:
        cf.commander.send_setpoint(0.0, 0.0, 0, thrust)
        time.sleep(0.02)  # 50 Hz


def _simple_hop(cf: Crazyflie):
    """Your thumbs_up pattern: hop with tuned thrust, soft-ish landing."""
    log.info("Drone: simple hop (thumbs_up pattern)")

    # 0) Unlock motors
    for _ in range(10):
        cf.commander.send_setpoint(0.0, 0.0, 0, 0)
        time.sleep(0.02)

    # 1) Ramp up
    log.info("Ramp up to hover thrust...")
    for t in range(20000, HOVER_THRUST + 1, 2000):
        log.info("Thrust ramp: %d", t)
        _send_thrust(cf, t, 0.06)

    # 2) Hover-ish
    log.info("Hover at %d for %.2fs", HOVER_THRUST, HOVER_TIME)
    _send_thrust(cf, HOVER_THRUST, HOVER_TIME)

    # 3) Controlled descent
    descent_thrust = int(HOVER_THRUST * DESCENT_FACTOR)
    log.info("Controlled descent at %d", descent_thrust)
    _send_thrust(cf, descent_thrust, 0.8)

    # 4) Final ramp down
    log.info("Final ramp down...")
    for t in range(descent_thrust, 16000, -2000):
        log.info("Thrust ramp down: %d", t)
        _send_thrust(cf, t, 0.08)

    # 5) Motors off
    cf.commander.send_setpoint(0.0, 0.0, 0, 0)
    cf.commander.send_stop_setpoint()
    log.info("Simple hop done")


def _run_flight(flight_fn):
    """
    Connect → run flight_fn(cf) → disconnect.
    Runs in a background thread and ensures only one flight at a time.
    """

    def worker(fn):
        global _is_flying
        try:
            with _flight_lock:
                if _is_flying:
                    log.info("Drone: already flying, ignoring new command")
                    return
                _is_flying = True

            log.info("Initializing Crazyflie drivers...")
            cflib.crtp.init_drivers(enable_debug_driver=False)

            log.info("Connecting to Crazyflie at URI: %s", CF_URI)
            cf = Crazyflie(rw_cache="./cf_cache")

            with SyncCrazyflie(CF_URI, cf=cf) as scf:
                log.info("Drone: connected")
                cf = scf.cf

                try:
                    fn(cf)
                finally:
                    try:
                        cf.commander.send_setpoint(0.0, 0.0, 0, 0)
                        cf.commander.send_stop_setpoint()
                    except Exception:
                        pass

        except Exception:
            log.exception("Error during Crazyflie flight")
        finally:
            _is_flying = False
            log.info("Drone: flight sequence finished")

    # start background thread
    threading.Thread(target=worker, args=(flight_fn,), daemon=True).start()


def handle_gesture(label: str):
    """
    Public API: call this from your gesture callback.

    label is e.g. "thumbs_up", "thumbs_down", "neutral".
    """
    log.info("Drone controller got gesture: %s", label)

    if label == "thumbs_up":
        _run_flight(_simple_hop)
    elif label == "thumbs_down":
        log.info("Thumbs down: no drone action")
    elif label == "neutral":
        log.info("Neutral: no drone action")
    else:
        log.warning("Unknown gesture label for drone: %s", label)
