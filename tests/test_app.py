import os

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(__file__), "..", "app.py")


def test_app_trains_on_sample_dataset():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    train = next(b for b in at.button if b.label == "Train model")
    train.click().run()
    assert not at.exception
    assert at.session_state["rec"] is not None
    assert at.session_state["report"]["accuracy"] > 0.9
