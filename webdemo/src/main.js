'use strict';

import { ImageInferenceHandler } from "./inference";
import { StatusDisplay } from "./status";

function initMap() {
  var map = L.map('output-map').setView([51.505, -0.09], 9);
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  }).addTo(map);
  return map;
}
const map = initMap();


const imgHandler = new ImageInferenceHandler(
  document.getElementById("input-image"),
  document.getElementById("input-image-canvas"),
  map,
  new StatusDisplay("status-display"),
);

function inputImageChanged(e) {
  const file = e.target.files[0];
  imgHandler.processUserFile(file);
}

document.getElementById("image-file")
  .addEventListener("change", inputImageChanged);


function generateExamples() {
  const e1 = new URL("sv__0IFyG3VFWd7XfHiy1LY9w.jpg", import.meta.url);
  /*const examplesArray = [
    [new URL("sv__0IFyG3VFWd7XfHiy1LY9w.jpg", import.meta.url), "Italy (near Naples)"],
  ];*/
  const examplesArray = [
    [e1, "Italy (near Naples)"],
  ];

  const exampleContainer = document.getElementById("examples");

  for (const exampleTuple of examplesArray) {
    const [url, captionText] = exampleTuple;

    const unit = document.createElement("div");
    unit.classList.add("pure-u-1");
    unit.classList.add("pure-u-md-1-3");

    const img = document.createElement("img");
    img.src = url;
    img.addEventListener("click", () => {
      imgHandler.processUrl(url);
    });
    unit.appendChild(img);

    const caption = document.createElement("div");
    caption.classList.add("caption");
    caption.textContent = captionText;
    unit.appendChild(caption);

    exampleContainer.appendChild(unit);
  }
}
generateExamples();