/*
  Small browser helpers for the OMR app.

  There is no frontend framework here on purpose. The app needs only two pieces
  of interactivity:
  1. Show the right number of answer-key rows when a teacher creates a test.
  2. Let a phone camera capture a sheet image and submit it to Flask.
*/

function setupAnswerKeyForm() {
  const form = document.querySelector("[data-answer-key-form]");

  if (!form) {
    return;
  }

  const countInput = form.querySelector("[data-question-count]");
  const rows = Array.from(form.querySelectorAll("[data-answer-row]"));

  function updateVisibleRows() {
    const count = Number.parseInt(countInput.value, 10) || 20;

    rows.forEach((row) => {
      const questionNumber = Number.parseInt(row.dataset.answerRow, 10);
      row.hidden = questionNumber > count;
    });
  }

  countInput.addEventListener("input", updateVisibleRows);
  updateVisibleRows();
}

function setupCameraCapture() {
  const form = document.querySelector("[data-scan-form]");

  if (!form) {
    return;
  }

  const video = form.querySelector("[data-camera-video]");
  const canvas = form.querySelector("[data-camera-canvas]");
  const preview = form.querySelector("[data-camera-preview]");
  const hiddenImage = form.querySelector("[data-captured-image]");
  const fileInput = form.querySelector('input[name="sheet_image"]');
  const compressCheckbox = form.querySelector("[data-compress-image]");
  const startButton = form.querySelector("[data-start-camera]");
  const captureButton = form.querySelector("[data-capture-photo]");
  const status = form.querySelector("[data-camera-status]");

  let stream = null;
  let submittingCompressedFile = false;

  const compressedLongSide = 1100;
  const compressedQuality = 0.72;
  const originalQuality = 0.95;

  function shouldCompress() {
    return !compressCheckbox || compressCheckbox.checked;
  }

  async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      status.textContent = "This browser does not support direct camera capture. Use the file chooser below.";
      return;
    }

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1280 },
          height: { ideal: 1920 },
        },
        audio: false,
      });

      video.srcObject = stream;
      video.hidden = false;
      preview.hidden = true;
      captureButton.disabled = false;
      status.textContent = "Camera is ready. Keep all four coded corner markers inside the photo.";
    } catch (error) {
      status.textContent = "Camera access failed. Use the file chooser below or open the app through the ngrok HTTPS link.";
    }
  }

  function capturePhoto() {
    if (!video.videoWidth || !video.videoHeight) {
      status.textContent = "Camera is not ready yet.";
      return;
    }

    const dataUrl = drawImageToDataUrl(
      video,
      video.videoWidth,
      video.videoHeight,
      shouldCompress()
    );

    hiddenImage.value = dataUrl;
    preview.src = dataUrl;
    preview.hidden = false;
    video.hidden = true;

    if (fileInput) {
      fileInput.disabled = true;
    }

    const mode = shouldCompress() ? "compressed" : "captured at full camera preview size";
    status.textContent = `Photo ${mode}. Upload size is about ${estimateDataUrlKilobytes(dataUrl)} KB. Press Process Scan.`;
  }

  function drawImageToDataUrl(source, sourceWidth, sourceHeight, compress) {
    /*
      Compression is recommended because phone images can be many megabytes and
      ngrok may reject the request before Flask receives it. When compression is
      off, camera captures still pass through canvas because getUserMedia gives
      us a video frame, not an original camera file.
    */
    const maxLongSide = compress ? compressedLongSide : Math.max(sourceWidth, sourceHeight);
    const quality = compress ? compressedQuality : originalQuality;
    const scale = Math.min(1, maxLongSide / Math.max(sourceWidth, sourceHeight));
    const width = Math.round(sourceWidth * scale);
    const height = Math.round(sourceHeight * scale);

    canvas.width = width;
    canvas.height = height;

    const context = canvas.getContext("2d");
    context.drawImage(source, 0, 0, width, height);

    return canvas.toDataURL("image/jpeg", quality);
  }

  function estimateDataUrlKilobytes(dataUrl) {
    const base64 = dataUrl.split(",")[1] || "";
    return Math.round((base64.length * 3) / 4 / 1024);
  }

  function compressFileToDataUrl(file) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      const objectUrl = URL.createObjectURL(file);

      image.onload = () => {
        try {
          const dataUrl = drawImageToDataUrl(
            image,
            image.naturalWidth,
            image.naturalHeight,
            true
          );
          URL.revokeObjectURL(objectUrl);
          resolve(dataUrl);
        } catch (error) {
          URL.revokeObjectURL(objectUrl);
          reject(error);
        }
      };

      image.onerror = () => {
        URL.revokeObjectURL(objectUrl);
        reject(new Error("Could not load the selected image."));
      };

      image.src = objectUrl;
    });
  }

  startButton.addEventListener("click", startCamera);
  captureButton.addEventListener("click", capturePhoto);

  if (fileInput) {
    fileInput.addEventListener("change", () => {
      hiddenImage.value = "";
      preview.hidden = true;
      fileInput.disabled = false;

      if (fileInput.files && fileInput.files.length > 0) {
        status.textContent = shouldCompress()
          ? "Image selected. It will be compressed before upload."
          : "Image selected. Original file will be uploaded.";
      }
    });
  }

  if (compressCheckbox) {
    compressCheckbox.addEventListener("change", () => {
      const message = shouldCompress()
        ? "Compression is on. This is recommended for ngrok."
        : "Compression is off. Large phone photos may be rejected by ngrok.";
      status.textContent = message;
    });
  }

  form.addEventListener("submit", async (event) => {
    if (submittingCompressedFile) {
      return;
    }

    if (hiddenImage.value) {
      if (fileInput) {
        fileInput.disabled = true;
      }
      return;
    }

    if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
      return;
    }

    if (!shouldCompress()) {
      return;
    }

    event.preventDefault();
    status.textContent = "Compressing image before upload...";

    try {
      const dataUrl = await compressFileToDataUrl(fileInput.files[0]);
      hiddenImage.value = dataUrl;
      preview.src = dataUrl;
      preview.hidden = false;
      video.hidden = true;
      fileInput.disabled = true;
      submittingCompressedFile = true;
      status.textContent = `Image compressed to about ${estimateDataUrlKilobytes(dataUrl)} KB. Uploading...`;
      form.submit();
    } catch (error) {
      status.textContent = "Could not compress the selected image. Turn compression off or take a new photo.";
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  setupAnswerKeyForm();
  setupCameraCapture();
});
