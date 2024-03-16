    // MODEL LINK
    const URL = "http://127.0.0.1:8080/Models/A_F-model/";
    const videoElement = document.getElementsByClassName('input_video')[0] ;
    const canvasElement = document.getElementsByClassName('output_canvas')[0];
    const canvasCtx = canvasElement.getContext('2d');
    let labelContainer = document.getElementById("label-container");


    let model, webcam, maxPredictions;

    // calling holistic api from mediapipe cdn
    const holistic = new Holistic({locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
    }});
    // console.log("Holistic loaded")

      // Load the image model and setup the webcam
    async function init() {
        const modelURL = URL + "model.json";
        const metadataURL = URL + "metadata.json";

        // load the model and metadata
        model = await tmImage.load(modelURL, metadataURL);
        maxPredictions = model.getTotalClasses();
        // console.log("model loaded")
        
        // append elements to the DOM
        // document.getElementById("webcam-container").appendChild(canvasElement);
        // labelContainer = document.getElementById("label-container");
        for (let i = 0; i < maxPredictions; i++) { // and class labels
            labelContainer.appendChild(document.createElement("div"));
        }

        // configuring our holistic api according to webcam and requirements
        holistic.setOptions({
          modelComplexity: 1,
          smoothLandmarks: true,
          enableSegmentation: true,
          smoothSegmentation: true,
          refineFaceLandmarks: true,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5
        });
  
        // creating an camera using holistic api 
        const camera = new Camera(videoElement, {
          onFrame: async () => {
            await holistic.send({image: videoElement});
          },
          width: 1280,
          height: 720
        });

        camera.start();
        // console.log("Camera element created")

        // on detecting webcam image draw landmarks
        window.requestAnimationFrame(loop);
        // holistic.onResults(draw); //change
      }

    async function loop() {
        // webcam.update(); // update the webcam frame
        holistic.onResults(draw);
        await predict();
        window.requestAnimationFrame(loop);
    }

    // run the webcam image through the image model
    async function predict() {
      // predict can take in an image, video or canvas html element
      const prediction = await model.predict(canvasElement);
      for (let i = 0; i < maxPredictions; i++) {
          const classPrediction =
              prediction[i].className + ": " + prediction[i].probability.toFixed(2);
          labelContainer.childNodes[i].innerHTML = classPrediction;
      }
    }

    function draw(results) {
      // console.log("Called draw function")
      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    
      // Only overwrite missing pixels.
      canvasCtx.globalCompositeOperation = 'destination-atop';
      canvasCtx.drawImage(
          results.image, 0, 0, canvasElement.width, canvasElement.height);
    
      canvasCtx.globalCompositeOperation = 'source-over';
      //drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS,
        //              {color: '#00FF00', lineWidth: 4});
      //drawLandmarks(canvasCtx, results.poseLandmarks,
      //              {color: '#FF0000', lineWidth: 2});
      //drawConnectors(canvasCtx, results.faceLandmarks, FACEMESH_TESSELATION,
       //               {color: '#C0C0C070', lineWidth: 1});
      drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS,
                      {color: '#CC0000', lineWidth: 5});
      drawLandmarks(canvasCtx, results.leftHandLandmarks,
                    {color: '#00FF00', lineWidth: 2});
      drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS,
                      {color: '#00CC00', lineWidth: 5});
      drawLandmarks(canvasCtx, results.rightHandLandmarks,
                    {color: '#FF0000', lineWidth: 2});
      canvasCtx.restore();
    }
    

