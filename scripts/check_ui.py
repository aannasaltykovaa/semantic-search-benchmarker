"""Exercise real Streamlit interactions without needing a browser."""
from streamlit.testing.v1 import AppTest
from src.search import ROOT

app = AppTest.from_file(str(ROOT/'src/app.py'), default_timeout=180).run()
assert not app.exception and not app.error
app.button[0].click().run()
assert not app.exception and len(app.expander) > 0
app.button[1].click().run()
assert not app.exception and len(app.dataframe) >= 2
print('PASS: initialization, live search, benchmark tables and charts')
