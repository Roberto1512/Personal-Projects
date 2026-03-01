# tests/test_behavioral_model.py

import pytest

from naplace.modeling.predict import predict_component_lstm


# -----------------------------
# Fixture: predictor riutilizzabile
# -----------------------------
@pytest.fixture(scope="module")
def predict():
    """
    Fixture che ritorna una funzione comoda per predire una lista di testi
    e ottenere una lista di label.
    """
    def _predict(texts: list[str]) -> list[str]:
        return predict_component_lstm(texts)

    return _predict


# -----------------------------
# 1) Invariance Tests
# -----------------------------
def test_invariance_whitespace_and_case(predict):
    """
    Cambiamenti superficiali (spazi, maiuscole/minuscole) non dovrebbero
    cambiare la predizione.
    """
    t1 = "Crash when opening the settings panel in Firefox."
    t2 = "  crash when opening the settings panel in firefox   "

    labels = predict([t1, t2])

    assert len(labels) == 2
    # Invariance: stessa label attesa
    assert labels[0] == labels[1]


def test_invariance_small_rewording(predict):
    """
    Piccole parafrasi che descrivono lo stesso bug dovrebbero avere la stessa label.
    (Se il test dovesse fallire, è un'indicazione di limite del modello, non un bug del codice.)
    """
    t1 = "When I open the Add-ons page, the browser crashes after a few seconds."
    t2 = "Opening the extensions (Add-ons) tab makes the browser crash shortly after."

    labels = predict([t1, t2])

    assert len(labels) == 2
    assert labels[0] == labels[1]
# -----------------------------
# 2) Directional Tests
# -----------------------------
@pytest.mark.xfail(reason="Current model tends to map different bug types to 'General' component.")
def test_directional_ui_vs_network(predict):
    """
    Due descrizioni di bug molto diverse dovrebbero portare a label diverse.
    Se il modello sbaglia e le mette uguali, il test evidenzia un comportamento
    poco desiderabile.
    """
    ui_bug = "The toolbar icons are misaligned and the settings button overlaps the menu."
    network_bug = "Page load fails when connected through a proxy, with DNS resolution errors."

    labels = predict([ui_bug, network_bug])

    assert len(labels) == 2
    # Directional: ci aspetteremmo component diversi
    assert labels[0] != labels[1]


@pytest.mark.xfail(reason="Current model tends to map different bug types to 'General' component.")
def test_directional_crash_vs_layout(predict):
    crash_bug = "Firefox crashes immediately when opening a new private window."
    layout_bug = "The page layout is broken on high DPI screens, with overlapping text."

    labels = predict([crash_bug, layout_bug])

    assert len(labels) == 2
    assert labels[0] != labels[1]

# -----------------------------
# 3) Minimum Functionality Tests
# -----------------------------
@pytest.mark.parametrize(
    "text",
    [
        "Browser crashes when opening the bookmarks sidebar.",
        "Console shows a JavaScript error when clicking the login button.",
        "Images fail to load on HTTPS sites but work on HTTP.",
    ],
)
def test_minimum_functionality_basic_predictions_not_empty(predict, text):
    """
    Il modello dovrebbe almeno produrre una label non vuota per alcuni casi base.
    """
    labels = predict([text])

    assert len(labels) == 1
    label = labels[0]

    assert label is not None
    assert isinstance(label, str)
    assert label.strip() != ""

@pytest.mark.xfail(
    reason="Current model tends to collapse crash + stacktrace and layout bugs into 'General'. "
           "Desiderata: distinguerli in categorie diverse."
)
def test_directional_stacktrace_detection(predict):
    text_with_stack = (
        "Browser crashes with invalid page fault.\n"
        "EAX=00000038 CS=0137 EIP=00436709 EFLGS=00010202\n"
        'Stack dump follows...'
    )

    no_stack = "Page layout overlaps text on 4K screens."

    labels = predict([text_with_stack, no_stack])

    # Spec desiderata: due label diverse
    assert labels[0] != labels[1], \
        "Stacktrace bug should not share label with a layout UI bug"

@pytest.mark.xfail(
    reason="Current model spesso assegna 'General' sia a bug Mail/Thunderbird che a bug Firefox UI. "
           "Desiderata: separare almeno product/ambito."
)
def test_directional_product_component_consistency(predict):
    mail_bug = "When sending an email, Thunderbird fails to encode attachments correctly."
    generic_bug = "Firefox UI freezes after opening a new tab."

    labels = predict([mail_bug, generic_bug])

    assert labels[0] != labels[1], (
        "A Mail/Composer bug should never map to the same label as a Firefox UI bug"
    )

def test_invariance_spam_comments(predict):
    base = "Firefox crashes on startup."
    noisy = (
        "Firefox crashes on startup.\n"
        "SPAM COMMENT: this bug is first!!\n"
        "off-topic\n"
    )

    labels = predict([base, noisy])
    assert labels[0] == labels[1], "Spam/noisy comments should not alter classification"

def test_invariance_multilingual(predict):
    en = "The page layout is broken on high DPI screens."
    it = "Il layout della pagina è rotto sugli schermi ad alta DPI."

    labels = predict([en, it])
    assert labels[0] == labels[1], "Multilingual equivalents should keep the same label"

@pytest.mark.xfail(
    reason="Current model tende a classificare come 'General' sia bug complessi che semplici glitch UI. "
           "Desiderata: differenziare almeno casi 'storici'/complessi dai trivial."
)
def test_directional_history_length(predict):
    long_history_bug = (
        "This bug has been reopened and reassigned many times. "
        "The keybindings do not work properly."
    )
    simple_bug = "Minor UI glitch when resizing window."

    labels = predict([long_history_bug, simple_bug])

    assert labels[0] != labels[1], "Complex historical bugs should not match simple UI glitches"
