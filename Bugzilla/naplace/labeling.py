# naplace/labeling.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Copiamo / ispiriamoci dalle costanti di ComponentModel
BUGBUG_PRODUCTS = {
    "Core",
    "External Software Affecting Firefox",
    "DevTools",
    "Firefox for Android",
    "Firefox",
    "Toolkit",
    "WebExtensions",
    "Firefox Build System",
}

CONFLATED_COMPONENTS = [
    "Core::Audio/Video",
    "Core::DOM",
    "Core::Graphics",
    "Core::IPC",
    "Core::JavaScript",
    "Core::Layout",
    "Core::Networking",
    "Core::Print",
    "Core::WebRTC",
    "Toolkit::Password Manager",
    "DevTools",
    "External Software Affecting Firefox",
    "WebExtensions",
    "Firefox Build System",
    "Firefox for Android",
]

CONFLATED_COMPONENTS_MAPPING = {
    "Core::DOM": "Core::DOM: Core & HTML",
    "Core::JavaScript": "Core::JavaScript Engine",
    "Core::Print": "Core::Printing: Output",
    "DevTools": "DevTools::General",
    "External Software Affecting Firefox": "External Software Affecting Firefox::Other",
    "WebExtensions": "WebExtensions::Untriaged",
    "Firefox Build System": "Firefox Build System::General",
    "Firefox for Android": "Firefox for Android::General",
}

# Inversa, come in ComponentModel
CONFLATED_COMPONENTS_INVERSE_MAPPING = {v: k for k, v in CONFLATED_COMPONENTS_MAPPING.items()}

# Componenti che vogliamo considerare "non informative"
TRASH_COMPONENTS = {"General", "Untriaged", "Foxfooding"}


@dataclass
class BugLabel:
    """Etichette che vogliamo assegnare ad ogni bug."""

    component_label: Optional[str]  # livello 2 (conflated / full)
    macro_component: Optional[str]  # livello 1 (Core, Firefox, Toolkit, ...)


def is_meaningful(product: str, component: str) -> bool:
    """Replica in modo semplificato la is_meaningful di BugBug."""
    if product in BUGBUG_PRODUCTS and component not in TRASH_COMPONENTS:
        return True

    # qui potresti aggiungere ulteriori prodotti speciali (Cloud Services, ecc.)
    return False


def map_bug_to_component_label(product: str, component: str) -> Optional[str]:
    """
    Approx di ComponentModel.filter_component + get_labels, ma offline.

    Ritorna:
      - una stringa tipo "Core::DOM" / "Toolkit::Password Manager" / "DevTools"
      - oppure None se il bug non è etichettabile secondo le regole
    """
    product = (product or "").strip()
    component = (component or "").strip()

    if not product or not component:
        return None

    # Se il prodotto non è tra quelli che ci interessano → scartiamo
    if not is_meaningful(product, component):
        return None

    full_comp = f"{product}::{component}"

    # Caso 1: il full_comp è uno di quelli "espansi" di CONFLATED_COMPONENTS_MAPPING
    if full_comp in CONFLATED_COMPONENTS_INVERSE_MAPPING:
        return CONFLATED_COMPONENTS_INVERSE_MAPPING[full_comp]

    # Caso 2: il full_comp è un componente "normale" (esiste come label sensata)
    # Qui non abbiamo get_meaningful_product_components, quindi usiamo una logica semplificata:
    # - se product è nella lista di prodotti validi e il componente non è "General"/"Untriaged"/"Foxfooding",
    #   teniamo il full_comp.
    if is_meaningful(product, component):
        return full_comp

    # Caso 3: fallback "conflated" se inizia con uno dei prefix conflati
    for conflated_component in CONFLATED_COMPONENTS:
        if full_comp.startswith(conflated_component):
            return conflated_component

    return None


def macro_from_component_label(component_label: str) -> Optional[str]:
    """
    Livello 1: da una label tipo "Core::DOM" / "DevTools" → macro categoria.
    """
    if not component_label:
        return None

    # Se è tipo "Core::Audio/Video" → macro è "Core"
    if "::" in component_label:
        macro = component_label.split("::", 1)[0]
        if macro in BUGBUG_PRODUCTS:
            return macro

    # Label "singole" che corrispondono già a una macro (DevTools, WebExtensions, ecc.)
    if component_label in BUGBUG_PRODUCTS:
        return component_label

    # Caso speciale: conflated component tipo "Core::DOM" (prima parte è macro)
    for prod in BUGBUG_PRODUCTS:
        if component_label.startswith(f"{prod}::"):
            return prod

    return None


def label_bug(product: str, component: str) -> BugLabel:
    """Convenience: ritorna entrambe le etichette (livello 1 e 2)."""
    comp_label = map_bug_to_component_label(product, component)
    macro = macro_from_component_label(comp_label) if comp_label else None
    return BugLabel(component_label=comp_label, macro_component=macro)
