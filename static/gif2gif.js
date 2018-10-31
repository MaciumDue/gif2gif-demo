const DEBUG = false;
const SIZE = 256;
const CAT_MODEL_URL = "edges2cats_AtoB.pict";

// model
var weights_cache = {}
function fetch_weights(path, progress_cb) {
  return new Promise(function(resolve, reject) {
    if (path in weights_cache) {
      resolve(weights_cache[path])
      return
    }

    var xhr = new XMLHttpRequest()
    xhr.open("GET", path, true)
    xhr.responseType = "arraybuffer"

    xhr.onprogress = function(e) {
      progress_cb(e.loaded, e.total)
    }

    xhr.onload = function(e) {
      if (xhr.status != 200) {
        reject("missing model")
        return
      }
      var buf = xhr.response
      if (!buf) {
        reject("invalid arraybuffer")
        return
      }

      var parts = []
      var offset = 0
      while (offset < buf.byteLength) {
        var b = new Uint8Array(buf.slice(offset, offset+4))
        offset += 4
        var len = (b[0] << 24) + (b[1] << 16) + (b[2] << 8) + b[3]
        parts.push(buf.slice(offset, offset + len))
        offset += len
      }

      var shapes = JSON.parse((new TextDecoder("utf8")).decode(parts[0]))
      var index = new Float32Array(parts[1])
      var encoded = new Uint8Array(parts[2])

      // decode using index
      var arr = new Float32Array(encoded.length)
      for (var i = 0; i < arr.length; i++) {
        arr[i] = index[encoded[i]]
      }

      var weights = {}
      var offset = 0
      for (var i = 0; i < shapes.length; i++) {
        var shape = shapes[i].shape
        var size = shape.reduce((total, num) => total * num)
        var values = arr.slice(offset, offset+size)
        var dlarr = dl.Array1D.new(values, "float32")
        weights[shapes[i].name] = dlarr.reshape(shape)
        offset += size
      }
      weights_cache[path] = weights
      resolve(weights)
    }
    xhr.send(null)
  })
}


function model(input, weights) {
  const math = dl.ENV.math

  function preprocess(input) {
    return math.subtract(math.multiply(input, dl.Scalar.new(2)), dl.Scalar.new(1))
  }

  function deprocess(input) {
    return math.divide(math.add(input, dl.Scalar.new(1)), dl.Scalar.new(2))
  }

  function batchnorm(input, scale, offset) {
    var moments = math.moments(input, [0, 1])
    const varianceEpsilon = 1e-5
    return math.batchNormalization3D(input, moments.mean, moments.variance, varianceEpsilon, scale, offset)
  }

  function conv2d(input, filter, bias) {
    return math.conv2d(input, filter, bias, [2, 2], "same")
  }

  function deconv2d(input, filter, bias) {
    var convolved = math.conv2dTranspose(input, filter, [input.shape[0]*2, input.shape[1]*2, filter.shape[2]], [2, 2], "same")
    var biased = math.add(convolved, bias)
    return biased
  }

  var preprocessed_input = preprocess(input)

  var layers = []

  var filter = weights["generator/encoder_1/conv2d/kernel"]
  var bias = weights["generator/encoder_1/conv2d/bias"]
  var convolved = conv2d(preprocessed_input, filter, bias)
  layers.push(convolved)

  for (var i = 2; i <= 8; i++) {
    var scope = "generator/encoder_" + i.toString()
    var filter = weights[scope + "/conv2d/kernel"]
    var bias = weights[scope + "/conv2d/bias"]
    var layer_input = layers[layers.length - 1]
    var rectified = math.leakyRelu(layer_input, 0.2)
    var convolved = conv2d(rectified, filter, bias)
    var scale = weights[scope + "/batch_normalization/gamma"]
    var offset = weights[scope + "/batch_normalization/beta"]
    var normalized = batchnorm(convolved, scale, offset)
    layers.push(normalized)
  }

  for (var i = 8; i >= 2; i--) {
    if (i == 8) {
      var layer_input = layers[layers.length - 1]
    } else {
      var skip_layer = i - 1
      var layer_input = math.concat3D(layers[layers.length - 1], layers[skip_layer], 2)
    }
    var rectified = math.relu(layer_input)
    var scope = "generator/decoder_" + i.toString()
    var filter = weights[scope + "/conv2d_transpose/kernel"]
    var bias = weights[scope + "/conv2d_transpose/bias"]
    var convolved = deconv2d(rectified, filter, bias)
    var scale = weights[scope + "/batch_normalization/gamma"]
    var offset = weights[scope + "/batch_normalization/beta"]
    var normalized = batchnorm(convolved, scale, offset)
    // missing dropout
    layers.push(normalized)
  }

  var layer_input = math.concat3D(layers[layers.length - 1], layers[0], 2)
  var rectified = math.relu(layer_input)
  var filter = weights["generator/decoder_1/conv2d_transpose/kernel"]
  var bias = weights["generator/decoder_1/conv2d_transpose/bias"]
  var convolved = deconv2d(rectified, filter, bias)
  var rectified = math.tanh(convolved)
  layers.push(rectified)

  var output = layers[layers.length - 1]
  var deprocessed_output = deprocess(output)

  return deprocessed_output
}


async function edges2image(modelUrl, imageDataUint8) {
  const modelWeights = await fetch_weights(modelUrl, () => null);

  var input_float32_data = Float32Array.from(imageDataUint8, (x) => x / 255)

  const math = dl.ENV.math
  math.startScope()
  var input_rgba = dl.Array3D.new([SIZE, SIZE, 4], input_float32_data, "float32")
  var input_rgb = math.slice3D(input_rgba, [0, 0, 0], [SIZE, SIZE, 3])

  var output_rgb = model(input_rgb, modelWeights)

  var alpha = dl.Array3D.ones([SIZE, SIZE, 1])
  var output_rgba = math.concat3D(output_rgb, alpha, 2)

  const output_float32_data = await output_rgba.getValuesAsync();
  var output_uint8_data = Uint8ClampedArray.from(output_float32_data, (x) => x * 255)
  math.endScope()
  return output_uint8_data;
}


function resizeCanvas(canvas, width, height) {
  const resizedCanvas = document.createElement('canvas');
  resizedCanvas.width = width;
  resizedCanvas.height = height;
  const ctx = resizedCanvas.getContext('2d');
  ctx.drawImage(canvas, 0, 0, width, height);
  return resizedCanvas;
}


function imageData2canvas(imageData) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = SIZE;
  const ctx = canvas.getContext('2d');
  ctx.putImageData(imageData, 0, 0);
  return canvas;
}

async function gif2inputs(gifUrl) {
  const frames = await gifFrames({ url: gifUrl, frames: 'all', outputType: 'canvas', cumulative: true});
  return frames.map((frameData) => resizeCanvas(frameData.getImage(), SIZE, SIZE));
}


function displayFramesAsGif(frames, writebackImageElement) {
  var gif = new GIF({
    workers: 2,
    quality: 1
  });

  frames.forEach(frame => gif.addFrame(frame, {delay: 200}));

  gif.on('finished', function(blob) {
    writebackImageElement.src = URL.createObjectURL(blob);
  });

  gif.render();
}


function canvasToUint8Data(canvas) {
  const ctx = canvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, SIZE, SIZE);
  return imageData.data;
}


async function frames2frames(frames) {
  const pix2pixFrameData = [];
  for (const frame of frames) {
    const generatedFrame = await edges2image(CAT_MODEL_URL, canvasToUint8Data(frame));
    pix2pixFrameData.push(generatedFrame);
  }
  return pix2pixFrameData;
}


function gif2gif() {
  const gifUrl = document.getElementById("input-gif-url").value;
  const inputGifImg = document.getElementById("input-gif");
  const outputGifImg = document.getElementById("generated-output-gif");

  gif2inputs(gifUrl).then((processedFrameData) => {
    displayFramesAsGif(processedFrameData, inputGifImg);
    frames2frames(processedFrameData).then(generatedFrames => {
      generatedFrames = generatedFrames.map(frameData => imageData2canvas(new ImageData(frameData, SIZE, SIZE)));
      displayFramesAsGif(generatedFrames, outputGifImg);
    });
  });
}
