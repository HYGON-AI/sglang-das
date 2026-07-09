import logging


logger = logging.getLogger(__name__)


def test_hcu_runtime_text_gate_failure_message():
    logger.warning("AMD GPU memory capacity detection failed.")
    logger.error("AMD 1 hop XGMI detection failed.")
    raise RuntimeError("DCU runtime reported an AMD GPU topology error.")
