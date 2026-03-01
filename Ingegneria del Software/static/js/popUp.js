document.addEventListener("DOMContentLoaded", function() {
  var popup = document.getElementById("popup");
  var background = document.getElementById("background");

  // Controlla se la variabile "checkout" è presente nel template
  var checkoutVariable = "{{ checkout }}";

  if (checkoutVariable) {
    openPopup();
  }

  // Funzione per aprire il pop-up
  function openPopup() {
    popup.style.display = "block";
    background.style.display = "block";
  }

  // Funzione per chiudere il pop-up
  function closePopup() {
    popup.style.display = "none";
    background.style.display = "none";
  }

  // Aggiungi l'evento di click al bottone per chiudere il pop-up
  var closeButton = document.querySelector("#popup button");
  closeButton.addEventListener("click", closePopup);
});
