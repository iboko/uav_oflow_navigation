"""Безопасность изоляции Gazebo и достоверность заявлений в отчете."""
import io
import signal

from tools.gz_failure_scenarios import Px4, new_report


def test_gazebo_report_does_not_claim_gnss_denied_navigation():
    report = new_report("ofnav_textured")
    assert report["validation_scope"] == "PX4_SITL_SAFETY_MODE_TRANSITIONS_ONLY"
    for field in ("gnss_denied_navigation_verified", "ekf2_ofnav_fusion_verified",
                  "ofnav_position_hold_verified", "real_aircraft_tested"):
        assert report[field] is False


def test_stop_targets_only_own_process_group(monkeypatch):
    instance = Px4.__new__(Px4)
    instance.log = io.StringIO()

    class Child:
        pid = 14321
        def wait(self, timeout):
            assert timeout == 10

    instance.proc = Child()
    instance.cmd = lambda *args, **kwargs: "shutdown complete"
    calls = []
    monkeypatch.setattr("tools.gz_failure_scenarios.os.killpg",
                        lambda pid, sig: calls.append((pid, sig)))
    def forbid_global_kill(*args, **kwargs):
        raise AssertionError("global pkill/subprocess.run is forbidden")
    monkeypatch.setattr("tools.gz_failure_scenarios.subprocess.run", forbid_global_kill)
    instance.stop()
    assert calls == [(14321, signal.SIGTERM)]
    assert instance.log.closed
