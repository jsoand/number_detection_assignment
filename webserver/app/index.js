const form = document.querySelector('form');
form.addEventListener('submit', handleSubmit);

let number = null;

function handleSubmit(event) {
  const form = event.currentTarget;
  const url = new URL("http://localhost:5000/detect");
  const formData = new FormData(form);
  const fetchOptions = {
    method: form.method,
    body: formData
  };

  fetch(url, fetchOptions).then(response => {
    response.json().then(data => {
      number = data;
      document.getElementById('detected-number').innerHTML = data;

      document.getElementById('number-response').style.visibility = 'visible';
    })
  });

  event.preventDefault();
}

function numberToText() {
  const url = new URL(`http://localhost:5001/number/${number}`);
  const fetchOptions = {
    method: "GET",
  };

  fetch(url, fetchOptions).then(response => {
    response.json().then(data => {
      document.getElementById('number-as-text').innerHTML = data;
      document.getElementById('t2s').style.visibility = 'visible';
    })
  });
}

function playAudio() {
  // TODO: implement
   // document.getElementById('translate').style.visibility = 'visible';

}

function translate() {
  // TODO: implement
}