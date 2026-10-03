(function () {
  "use strict";

  const FORMAT = "octo-mini-decoder-v1";
  const MODEL_DIMENSION = 16;
  const FEED_FORWARD_DIMENSION = 32;
  const CONTEXT_LENGTH = 32;
  const MAX_VOCABULARY_SIZE = 256;

  function zeros(length) {
    return Array.from({ length }, () => 0);
  }

  function matrix(rows, columns, random = false) {
    const scale = random ? Math.sqrt(2 / (rows + columns)) : 0;
    return Array.from({ length: rows }, () =>
      Array.from({ length: columns }, () => random ? (Math.random() * 2 - 1) * scale : 0),
    );
  }

  function matVec(weights, vector) {
    const result = zeros(weights[0].length);
    for (let row = 0; row < weights.length; row += 1) {
      const value = vector[row];
      for (let column = 0; column < result.length; column += 1) {
        result[column] += value * weights[row][column];
      }
    }
    return result;
  }

  function addInto(target, source) {
    for (let index = 0; index < target.length; index += 1) target[index] += source[index];
  }

  function addOuter(gradient, left, right) {
    for (let row = 0; row < left.length; row += 1) {
      for (let column = 0; column < right.length; column += 1) {
        gradient[row][column] += left[row] * right[column];
      }
    }
  }

  function softmax(values, temperature = 1) {
    const maximum = Math.max(...values);
    const exponents = values.map(value => Math.exp(Math.max(-40, (value - maximum) / temperature)));
    const total = exponents.reduce((sum, value) => sum + value, 0);
    return exponents.map(value => value / total);
  }

  function createModel(characters) {
    const frequencies = new Map();
    for (const character of characters) {
      frequencies.set(character, (frequencies.get(character) || 0) + 1);
    }
    const priorityCharacters = [" ", "\n", "\t", ":", "?", "!", ".", ","];
    const vocabulary = priorityCharacters.filter(character => frequencies.has(character));
    const remaining = [...frequencies.entries()]
      .filter(([character]) => !vocabulary.includes(character))
      .sort((left, right) => right[1] - left[1])
      .slice(0, MAX_VOCABULARY_SIZE - vocabulary.length)
      .map(([character]) => character);
    vocabulary.push(...remaining);
    if (vocabulary.length < 2) throw new Error("훈련 문장에 서로 다른 문자가 두 개 이상 필요합니다.");

    const dimension = MODEL_DIMENSION;
    const hidden = FEED_FORWARD_DIMENSION;
    return {
      format: FORMAT,
      language: "ko",
      tokenizer: "unicode-character",
      vocabulary,
      config: {
        dimension,
        hidden,
        context: CONTEXT_LENGTH,
        maxVocabulary: MAX_VOCABULARY_SIZE,
        layers: 1,
        heads: 1,
        objective: "next-character-prediction",
      },
      weights: {
        tokenEmbedding: matrix(vocabulary.length, dimension, true),
        positionEmbedding: matrix(CONTEXT_LENGTH, dimension, true),
        query: matrix(dimension, dimension, true),
        key: matrix(dimension, dimension, true),
        value: matrix(dimension, dimension, true),
        attentionOutput: matrix(dimension, dimension, true),
        feedForwardIn: matrix(dimension, hidden, true),
        feedForwardInBias: zeros(hidden),
        feedForwardOut: matrix(hidden, dimension, true),
        feedForwardOutBias: zeros(dimension),
        languageHead: matrix(dimension, vocabulary.length, true),
        languageHeadBias: zeros(vocabulary.length),
      },
    };
  }

  function validateModel(model) {
    if (!model || model.format !== FORMAT || model.language !== "ko") {
      throw new Error("지원하지 않는 모델 형식입니다.");
    }
    if (!Array.isArray(model.vocabulary) || model.vocabulary.length < 2 || model.vocabulary.some(token => typeof token !== "string" || [...token].length !== 1)) {
      throw new Error("모델의 문자 어휘가 올바르지 않습니다.");
    }
    if (new Set(model.vocabulary).size !== model.vocabulary.length) throw new Error("모델 어휘에 중복 토큰이 있습니다.");
    const config = model.config;
    const weights = model.weights;
    const d = MODEL_DIMENSION;
    const h = FEED_FORWARD_DIMENSION;
    const v = model.vocabulary.length;
    if (!config || config.dimension !== d || config.hidden !== h || config.context !== CONTEXT_LENGTH || config.maxVocabulary !== MAX_VOCABULARY_SIZE || config.layers !== 1 || config.heads !== 1) {
      throw new Error("지원하지 않는 모델 구조입니다.");
    }

    const matrixShapes = {
      tokenEmbedding: [v, d],
      positionEmbedding: [CONTEXT_LENGTH, d],
      query: [d, d],
      key: [d, d],
      value: [d, d],
      attentionOutput: [d, d],
      feedForwardIn: [d, h],
      feedForwardOut: [h, d],
      languageHead: [d, v],
    };
    for (const [name, shape] of Object.entries(matrixShapes)) {
      const value = weights && weights[name];
      if (!Array.isArray(value) || value.length !== shape[0] || value.some(row => !Array.isArray(row) || row.length !== shape[1] || row.some(number => !Number.isFinite(number)))) {
        throw new Error(`모델 가중치의 크기나 값이 올바르지 않습니다: ${name}`);
      }
    }
    const vectorShapes = { feedForwardInBias: h, feedForwardOutBias: d, languageHeadBias: v };
    for (const [name, length] of Object.entries(vectorShapes)) {
      const value = weights[name];
      if (!Array.isArray(value) || value.length !== length || value.some(number => !Number.isFinite(number))) {
        throw new Error(`모델 편향값의 크기나 값이 올바르지 않습니다: ${name}`);
      }
    }
    return model;
  }

  function forward(model, ids) {
    const { dimension: d, hidden: h } = model.config;
    const weights = model.weights;
    const length = ids.length;
    const states = ids.map((id, position) =>
      weights.tokenEmbedding[id].map((value, index) => value + weights.positionEmbedding[position][index]),
    );
    const queries = states.map(state => matVec(weights.query, state));
    const keys = states.map(state => matVec(weights.key, state));
    const values = states.map(state => matVec(weights.value, state));
    const attentions = [];
    const attentionStates = [];
    const residualStates = [];
    const hiddenStates = [];
    const outputs = [];
    const logits = [];

    for (let position = 0; position < length; position += 1) {
      const scores = [];
      for (let source = 0; source <= position; source += 1) {
        let score = 0;
        for (let index = 0; index < d; index += 1) score += queries[position][index] * keys[source][index];
        scores.push(score / Math.sqrt(d));
      }
      const attention = softmax(scores);
      attentions.push(attention);
      const attended = zeros(d);
      for (let source = 0; source <= position; source += 1) {
        for (let index = 0; index < d; index += 1) attended[index] += attention[source] * values[source][index];
      }
      attentionStates.push(attended);
      const attentionOutput = matVec(weights.attentionOutput, attended);
      const residual = states[position].map((value, index) => value + attentionOutput[index]);
      residualStates.push(residual);
      const hiddenState = matVec(weights.feedForwardIn, residual).map((value, index) =>
        Math.max(0, value + weights.feedForwardInBias[index]),
      );
      hiddenStates.push(hiddenState);
      const feedForwardOutput = matVec(weights.feedForwardOut, hiddenState);
      const output = residual.map((value, index) =>
        value + feedForwardOutput[index] + weights.feedForwardOutBias[index],
      );
      outputs.push(output);
      const tokenLogits = matVec(weights.languageHead, output).map((value, index) =>
        value + weights.languageHeadBias[index],
      );
      logits.push(tokenLogits);
    }

    return { ids, length, states, queries, keys, values, attentions, attentionStates, residualStates, hiddenStates, outputs, logits };
  }

  function gradientsFor(model) {
    const weights = model.weights;
    const gradients = {};
    for (const [name, value] of Object.entries(weights)) {
      gradients[name] = Array.isArray(value[0])
        ? value.map(row => zeros(row.length))
        : zeros(value.length);
    }
    return gradients;
  }

  function backpropagate(model, cache, targets) {
    const { dimension: d, hidden: h } = model.config;
    const weights = model.weights;
    const gradients = gradientsFor(model);
    const stateGradients = Array.from({ length: cache.length }, () => zeros(d));
    let loss = 0;
    const scale = 1 / targets.length;

    for (let position = 0; position < targets.length; position += 1) {
      const probabilities = softmax(cache.logits[position]);
      const target = targets[position];
      loss -= Math.log(Math.max(probabilities[target], 1e-12));
      probabilities[target] -= 1;
      for (let token = 0; token < probabilities.length; token += 1) {
        const gradient = probabilities[token] * scale;
        gradients.languageHeadBias[token] += gradient;
        for (let index = 0; index < d; index += 1) {
          gradients.languageHead[index][token] += cache.outputs[position][index] * gradient;
          stateGradients[position][index] += weights.languageHead[index][token] * gradient;
        }
      }
    }

    const attendedGradients = Array.from({ length: cache.length }, () => zeros(d));
    for (let position = 0; position < cache.length; position += 1) {
      const outputGradient = stateGradients[position];
      const hiddenGradient = zeros(h);
      addOuter(gradients.feedForwardOut, cache.hiddenStates[position], outputGradient);
      for (let index = 0; index < d; index += 1) {
        gradients.feedForwardOutBias[index] += outputGradient[index];
      }
      for (let hiddenIndex = 0; hiddenIndex < h; hiddenIndex += 1) {
        let value = 0;
        for (let index = 0; index < d; index += 1) {
          value += weights.feedForwardOut[hiddenIndex][index] * outputGradient[index];
        }
        hiddenGradient[hiddenIndex] = cache.hiddenStates[position][hiddenIndex] > 0 ? value : 0;
      }

      const residualGradient = outputGradient.slice();
      addOuter(gradients.feedForwardIn, cache.residualStates[position], hiddenGradient);
      for (let hiddenIndex = 0; hiddenIndex < h; hiddenIndex += 1) {
        gradients.feedForwardInBias[hiddenIndex] += hiddenGradient[hiddenIndex];
      }
      for (let index = 0; index < d; index += 1) {
        for (let hiddenIndex = 0; hiddenIndex < h; hiddenIndex += 1) {
          residualGradient[index] += weights.feedForwardIn[index][hiddenIndex] * hiddenGradient[hiddenIndex];
        }
      }

      stateGradients[position] = residualGradient;
      addOuter(gradients.attentionOutput, cache.attentionStates[position], residualGradient);
      for (let inputIndex = 0; inputIndex < d; inputIndex += 1) {
        let value = 0;
        for (let outputIndex = 0; outputIndex < d; outputIndex += 1) {
          value += weights.attentionOutput[inputIndex][outputIndex] * residualGradient[outputIndex];
        }
        attendedGradients[position][inputIndex] = value;
      }
    }

    const queryGradients = Array.from({ length: cache.length }, () => zeros(d));
    const keyGradients = Array.from({ length: cache.length }, () => zeros(d));
    const valueGradients = Array.from({ length: cache.length }, () => zeros(d));

    for (let position = 0; position < cache.length; position += 1) {
      for (let source = 0; source <= position; source += 1) {
        for (let index = 0; index < d; index += 1) {
          valueGradients[source][index] += cache.attentions[position][source] * attendedGradients[position][index];
        }
      }
    }

    for (let position = 0; position < cache.length; position += 1) {
      const attention = cache.attentions[position];
      const scoreGradients = [];
      let weightedScoreGradient = 0;
      for (let source = 0; source <= position; source += 1) {
        let gradient = 0;
        for (let index = 0; index < d; index += 1) {
          gradient += attendedGradients[position][index] * cache.values[source][index];
        }
        scoreGradients.push(gradient);
        weightedScoreGradient += attention[source] * gradient;
      }
      for (let source = 0; source <= position; source += 1) {
        const scoreGradient = attention[source] * (scoreGradients[source] - weightedScoreGradient) / Math.sqrt(d);
        for (let index = 0; index < d; index += 1) {
          queryGradients[position][index] += scoreGradient * cache.keys[source][index];
          keyGradients[source][index] += scoreGradient * cache.queries[position][index];
        }
      }
    }

    for (let position = 0; position < cache.length; position += 1) {
      addOuter(gradients.query, cache.states[position], queryGradients[position]);
      addOuter(gradients.key, cache.states[position], keyGradients[position]);
      addOuter(gradients.value, cache.states[position], valueGradients[position]);
      const inputGradient = stateGradients[position];
      for (let index = 0; index < d; index += 1) {
        for (let outputIndex = 0; outputIndex < d; outputIndex += 1) {
          inputGradient[index] +=
            weights.query[index][outputIndex] * queryGradients[position][outputIndex] +
            weights.key[index][outputIndex] * keyGradients[position][outputIndex] +
            weights.value[index][outputIndex] * valueGradients[position][outputIndex];
        }
      }
      addInto(gradients.tokenEmbedding[cache.ids[position]], inputGradient);
      addInto(gradients.positionEmbedding[position], inputGradient);
    }

    let squaredNorm = 0;
    for (const gradient of Object.values(gradients)) {
      const values = Array.isArray(gradient[0]) ? gradient.flat() : gradient;
      for (const value of values) squaredNorm += value * value;
    }
    const norm = Math.sqrt(squaredNorm);
    const clipScale = norm > 1 ? 1 / norm : 1;
    const learningRate = 0.08;
    for (const [name, gradient] of Object.entries(gradients)) {
      const weightsToUpdate = weights[name];
      if (Array.isArray(gradient[0])) {
        for (let row = 0; row < gradient.length; row += 1) {
          for (let column = 0; column < gradient[row].length; column += 1) {
            weightsToUpdate[row][column] -= learningRate * gradient[row][column] * clipScale;
          }
        }
      } else {
        for (let index = 0; index < gradient.length; index += 1) {
          weightsToUpdate[index] -= learningRate * gradient[index] * clipScale;
        }
      }
    }
    return loss / targets.length;
  }

  function trainStep(model, corpus, contextLength = CONTEXT_LENGTH) {
    validateModel(model);
    const tokenMap = new Map(model.vocabulary.map((token, index) => [token, index]));
    const segments = [];
    let currentSegment = [];
    for (const character of corpus) {
      const id = tokenMap.get(character);
      if (id === undefined) {
        if (currentSegment.length > 1) segments.push(currentSegment);
        currentSegment = [];
      } else {
        currentSegment.push(id);
      }
    }
    if (currentSegment.length > 1) segments.push(currentSegment);
    if (!segments.length) throw new Error("모델 어휘에 포함된 연속 문자가 두 개 이상인 훈련 구간을 찾지 못했습니다.");

    const ids = segments[Math.floor(Math.random() * segments.length)];
    const length = Math.min(contextLength, ids.length - 1);
    const start = Math.floor(Math.random() * (ids.length - length));
    const inputs = ids.slice(start, start + length);
    const targets = ids.slice(start + 1, start + length + 1);
    const cache = forward(model, inputs);
    return backpropagate(model, cache, targets);
  }

  function generate(model, prompt, maximumCharacters = 160, temperature = 0.8) {
    validateModel(model);
    const tokenMap = new Map(model.vocabulary.map((token, index) => [token, index]));
    const result = [...prompt].filter(character => tokenMap.has(character)).map(character => tokenMap.get(character));
    if (!result.length) result.push(Math.floor(Math.random() * model.vocabulary.length));

    for (let step = 0; step < maximumCharacters; step += 1) {
      const context = result.slice(-model.config.context);
      const cache = forward(model, context);
      const probabilities = softmax(cache.logits[cache.logits.length - 1], temperature);
      let choice = Math.random();
      let nextToken = probabilities.length - 1;
      for (let index = 0; index < probabilities.length; index += 1) {
        choice -= probabilities[index];
        if (choice <= 0) {
          nextToken = index;
          break;
        }
      }
      result.push(nextToken);
    }
    return result.map(id => model.vocabulary[id]).join("");
  }

  window.OCTO_MINI_GPT = Object.freeze({
    format: FORMAT,
    config: Object.freeze({
      dimension: MODEL_DIMENSION,
      hidden: FEED_FORWARD_DIMENSION,
      context: CONTEXT_LENGTH,
      maxVocabulary: MAX_VOCABULARY_SIZE,
      layers: 1,
      heads: 1,
    }),
    createModel,
    validateModel,
    trainStep,
    generate,
  });
})();
