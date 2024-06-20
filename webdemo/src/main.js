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
  const examplesArray = [
    [
      new URL("examples/sv__0IFyG3VFWd7XfHiy1LY9w.jpg", import.meta.url),
      "Italy (near Naples)"
    ],
    [
      new URL("examples/sv__djmpcK-Eofhl9xcPDdoLA.jpg", import.meta.url),
      "NYC (Hoboken)"
    ],
    [
      new URL("examples/sv__OMAtQ9zCwQGyVzrpxtxfg.jpg", import.meta.url),
      "Japan (near Fukuoka)"
    ],
    [
      new URL("examples/sv__PAgKQ38IE8PstRsbApzGQ.jpg", import.meta.url),
      "Japan (Tokyo)"
    ],
    [
      new URL("examples/sv_eP8H2JRDpYKVCck-vkw8gg.jpg", import.meta.url),
      "California, USA (near Stockton)"
    ],
    [
      new URL("examples/sv_eR0OxFSWO4p8gFNPneJFtQ.jpg", import.meta.url),
      "Switzerland (East region)"
    ],
    [
      new URL("examples/sv_oX6JMJhxRHPN5hS2X-y2rw.jpg", import.meta.url),
      "Germany (near Cologne)"
    ],
    [
      new URL("examples/sv_oZSwJfK_88z6bgUti7Ivzg.jpg", import.meta.url),
      "Indonesia (Central Java)"
    ],
    [
      new URL("examples/sv_u0XVxdurG6afYxiyBRf7jw.jpg", import.meta.url),
      "Nigeria (Lagos)"
    ]
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
      window.scrollTo(0, 0);
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