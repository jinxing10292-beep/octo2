(function () {
  "use strict";

  const FORMAT = "octo-mini-decoder-v3";
  const TOKENIZER = "ko-word-ending-char-v1";
  const MODEL_DIMENSION = 64;
  const ATTENTION_HEADS = 2;
  const HEAD_DIMENSION = MODEL_DIMENSION / ATTENTION_HEADS;
  const FEED_FORWARD_DIMENSION = MODEL_DIMENSION * 2;
  const TRANSFORMER_LAYERS = 3;
  const CONTEXT_LENGTH = 216;
  const MAX_VOCABULARY_SIZE = 1024;
  const WORD_PATTERN = /[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*/gu;
  const HANGUL_FINAL_CONSONANTS = [
    null, "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ",
    "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ",
    "ㅋ", "ㅌ", "ㅍ", "ㅎ",
  ];
  const validatedModels = new WeakSet();

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

  function getRegisteredEndings() {
    const groups = window.OCTO_KO_VOCABULARY?.endings?.groups || {};
    const endings = new Set();
    for (const values of Object.values(groups)) {
      for (const value of values.trim().split(/\s+/u)) {
        const ending = value.replace(/^-/, "").replace(/-$/, "");
        if (ending) endings.add(ending);
      }
    }
    return [...endings].sort((left, right) => [...right].length - [...left].length);
  }

  function createModel(corpus) {
    if (typeof corpus !== "string") throw new Error("훈련 문장은 문자열이어야 합니다.");
    const characterFrequencies = new Map();
    for (const character of corpus) {
      characterFrequencies.set(character, (characterFrequencies.get(character) || 0) + 1);
    }
    const wordFrequencies = new Map();
    for (const match of corpus.matchAll(WORD_PATTERN)) {
      wordFrequencies.set(match[0], (wordFrequencies.get(match[0]) || 0) + 1);
    }
    const endings = getRegisteredEndings();
    const priorityCharacters = [" ", "\n", "\t", ":", "?", "!", ".", ","];
    const sortedCharacters = [
      ...priorityCharacters.filter(character => characterFrequencies.has(character)),
      ...[...characterFrequencies.entries()]
        .filter(([character]) => !priorityCharacters.includes(character))
        .sort((left, right) => right[1] - left[1])
        .map(([character]) => character),
    ];
    const characterBudget = Math.min(
      sortedCharacters.length,
      MAX_VOCABULARY_SIZE - Math.min(256, endings.length + 16),
    );
    const vocabulary = sortedCharacters.slice(0, characterBudget);
    for (const ending of endings) {
      if (vocabulary.length >= MAX_VOCABULARY_SIZE) break;
      if (!vocabulary.includes(ending)) vocabulary.push(ending);
    }
    const sortedWords = [...wordFrequencies.entries()]
      .sort((left, right) => right[1] - left[1])
      .map(([word]) => word);
    for (const word of sortedWords) {
      if (vocabulary.length >= MAX_VOCABULARY_SIZE) break;
      if (!vocabulary.includes(word)) vocabulary.push(word);
    }
    if (vocabulary.length < 2) throw new Error("훈련 문장에 서로 다른 토큰이 두 개 이상 필요합니다.");

    const dimension = MODEL_DIMENSION;
    const hidden = FEED_FORWARD_DIMENSION;
    return {
      format: FORMAT,
      language: "ko",
      tokenizer: TOKENIZER,
      endings,
      vocabulary,
      config: {
        dimension,
        hidden,
        context: CONTEXT_LENGTH,
        maxVocabulary: MAX_VOCABULARY_SIZE,
        layers: TRANSFORMER_LAYERS,
        heads: ATTENTION_HEADS,
        objective: "next-token-prediction",
      },
      weights: {
        tokenEmbedding: matrix(vocabulary.length, dimension, true),
        positionEmbedding: matrix(CONTEXT_LENGTH, dimension, true),
        layers: Array.from({ length: TRANSFORMER_LAYERS }, () => ({
          query: matrix(dimension, dimension, true),
          key: matrix(dimension, dimension, true),
          value: matrix(dimension, dimension, true),
          attentionOutput: matrix(dimension, dimension, true),
          feedForwardIn: matrix(dimension, hidden, true),
          feedForwardInBias: zeros(hidden),
          feedForwardOut: matrix(hidden, dimension, true),
          feedForwardOutBias: zeros(dimension),
        })),
        languageHead: matrix(dimension, vocabulary.length, true),
        languageHeadBias: zeros(vocabulary.length),
      },
    };
  }

  function validateMatrix(value, rows, columns, name) {
    if (!Array.isArray(value) || value.length !== rows ||
        value.some(row => !Array.isArray(row) || row.length !== columns ||
          row.some(number => !Number.isFinite(number)))) {
      throw new Error(`모델 가중치의 크기나 값이 올바르지 않습니다: ${name}`);
    }
  }

  function validateVector(value, length, name) {
    if (!Array.isArray(value) || value.length !== length || value.some(number => !Number.isFinite(number))) {
      throw new Error(`모델 편향값의 크기나 값이 올바르지 않습니다: ${name}`);
    }
  }

  function validateExactKeys(value, expectedKeys, name) {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new Error(`${name} 구조가 올바르지 않습니다.`);
    }
    const actualKeys = Object.keys(value).sort();
    const expected = [...expectedKeys].sort();
    if (actualKeys.length !== expected.length || actualKeys.some((key, index) => key !== expected[index])) {
      throw new Error(`${name}에 알 수 없거나 누락된 항목이 있습니다.`);
    }
  }

  function validateModel(model) {
    if (!model || model.format !== FORMAT || model.language !== "ko") {
      throw new Error("지원하지 않는 가중치 형식입니다. 3층 모델 가중치 JSON인지 확인하세요.");
    }
    validateExactKeys(model, ["format", "language", "tokenizer", "endings", "vocabulary", "config", "weights"], "모델");
    if (!Array.isArray(model.vocabulary) || model.vocabulary.length < 2 ||
        model.vocabulary.length > MAX_VOCABULARY_SIZE ||
        model.vocabulary.some(token => typeof token !== "string" || !token)) {
      throw new Error(`모델 어휘는 비어 있지 않은 토큰 ${MAX_VOCABULARY_SIZE}개 이하여야 합니다.`);
    }
    if (new Set(model.vocabulary).size !== model.vocabulary.length) {
      throw new Error("모델 어휘에 중복 토큰이 있습니다.");
    }
    if (!Array.isArray(model.endings) || model.endings.length > 512 ||
        model.endings.some(ending => typeof ending !== "string" || !ending || [...ending].length > 32) ||
        new Set(model.endings).size !== model.endings.length) {
      throw new Error("모델 어미 목록이 올바르지 않습니다.");
    }

    const config = model.config;
    const weights = model.weights;
    const d = MODEL_DIMENSION;
    const h = FEED_FORWARD_DIMENSION;
    const v = model.vocabulary.length;
    validateExactKeys(
      config,
      ["dimension", "hidden", "context", "maxVocabulary", "layers", "heads", "objective"],
      "모델 설정",
    );
    if (config.dimension !== d || config.hidden !== h ||
        config.context !== CONTEXT_LENGTH || config.maxVocabulary !== MAX_VOCABULARY_SIZE ||
        config.layers !== TRANSFORMER_LAYERS || config.heads !== ATTENTION_HEADS ||
        config.objective !== "next-token-prediction" || model.tokenizer !== TOKENIZER) {
      throw new Error("가중치의 모델 설정이 현재 3층·2헤드 구조와 일치하지 않습니다.");
    }
    validateExactKeys(weights, ["tokenEmbedding", "positionEmbedding", "layers", "languageHead", "languageHeadBias"], "모델 가중치");

    validateMatrix(weights.tokenEmbedding, v, d, "tokenEmbedding");
    validateMatrix(weights.positionEmbedding, CONTEXT_LENGTH, d, "positionEmbedding");
    validateMatrix(weights.languageHead, d, v, "languageHead");
    validateVector(weights.languageHeadBias, v, "languageHeadBias");
    if (!Array.isArray(weights.layers) || weights.layers.length !== TRANSFORMER_LAYERS) {
      throw new Error(`모델에는 트랜스포머 층이 정확히 ${TRANSFORMER_LAYERS}개 있어야 합니다.`);
    }

    for (let layerIndex = 0; layerIndex < TRANSFORMER_LAYERS; layerIndex += 1) {
      const layer = weights.layers[layerIndex];
      const prefix = `layers[${layerIndex}]`;
      validateExactKeys(
        layer,
        ["query", "key", "value", "attentionOutput", "feedForwardIn", "feedForwardInBias", "feedForwardOut", "feedForwardOutBias"],
        prefix,
      );
      for (const name of ["query", "key", "value", "attentionOutput"]) {
        validateMatrix(layer[name], d, d, `${prefix}.${name}`);
      }
      validateMatrix(layer.feedForwardIn, d, h, `${prefix}.feedForwardIn`);
      validateVector(layer.feedForwardInBias, h, `${prefix}.feedForwardInBias`);
      validateMatrix(layer.feedForwardOut, h, d, `${prefix}.feedForwardOut`);
      validateVector(layer.feedForwardOutBias, d, `${prefix}.feedForwardOutBias`);
    }
    validatedModels.add(model);
    return model;
  }

  function forward(model, ids) {
    const d = MODEL_DIMENSION;
    const weights = model.weights;
    let states = ids.map((id, position) =>
      weights.tokenEmbedding[id].map((value, index) => value + weights.positionEmbedding[position][index]),
    );
    const layerCaches = [];

    for (const layerWeights of weights.layers) {
      const queries = states.map(state => matVec(layerWeights.query, state));
      const keys = states.map(state => matVec(layerWeights.key, state));
      const values = states.map(state => matVec(layerWeights.value, state));
      const attentions = [];
      const attentionStates = [];
      const residualStates = [];
      const hiddenStates = [];
      const outputs = [];

      for (let position = 0; position < states.length; position += 1) {
        const positionHeads = [];
        const attended = zeros(d);
        for (let head = 0; head < ATTENTION_HEADS; head += 1) {
          const headStart = head * HEAD_DIMENSION;
          const scores = [];
          for (let source = 0; source <= position; source += 1) {
            let score = 0;
            for (let index = headStart; index < headStart + HEAD_DIMENSION; index += 1) {
              score += queries[position][index] * keys[source][index];
            }
            scores.push(score / Math.sqrt(HEAD_DIMENSION));
          }
          const attention = softmax(scores);
          positionHeads.push(attention);
          for (let source = 0; source <= position; source += 1) {
            for (let index = headStart; index < headStart + HEAD_DIMENSION; index += 1) {
              attended[index] += attention[source] * values[source][index];
            }
          }
        }
        attentions.push(positionHeads);
        attentionStates.push(attended);

        const attentionOutput = matVec(layerWeights.attentionOutput, attended);
        const residual = states[position].map((value, index) => value + attentionOutput[index]);
        residualStates.push(residual);
        const hidden = matVec(layerWeights.feedForwardIn, residual).map((value, index) =>
          Math.max(0, value + layerWeights.feedForwardInBias[index]),
        );
        hiddenStates.push(hidden);
        const feedForwardOutput = matVec(layerWeights.feedForwardOut, hidden);
        const output = residual.map((value, index) =>
          value + feedForwardOutput[index] + layerWeights.feedForwardOutBias[index],
        );
        outputs.push(output);
      }

      layerCaches.push({ states, queries, keys, values, attentions, attentionStates, residualStates, hiddenStates, outputs });
      states = outputs;
    }

    const logits = states.map(state =>
      matVec(weights.languageHead, state).map((value, index) => value + weights.languageHeadBias[index]),
    );
    return { ids, states, layerCaches, logits };
  }

  function zerosLike(value) {
    if (Array.isArray(value)) return value.map(zerosLike);
    if (value && typeof value === "object") {
      return Object.fromEntries(Object.entries(value).map(([key, child]) => [key, zerosLike(child)]));
    }
    return 0;
  }

  function gradientsFor(model) {
    return zerosLike(model.weights);
  }

  function backpropagate(model, cache, targets) {
    const weights = model.weights;
    const gradients = gradientsFor(model);
    const d = MODEL_DIMENSION;
    const h = FEED_FORWARD_DIMENSION;
    const sequenceLength = cache.ids.length;
    const stateGradients = Array.from({ length: sequenceLength }, () => zeros(d));
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
          gradients.languageHead[index][token] += cache.states[position][index] * gradient;
          stateGradients[position][index] += weights.languageHead[index][token] * gradient;
        }
      }
    }

    for (let layerIndex = TRANSFORMER_LAYERS - 1; layerIndex >= 0; layerIndex -= 1) {
      const layerWeights = weights.layers[layerIndex];
      const layerGradients = gradients.layers[layerIndex];
      const layerCache = cache.layerCaches[layerIndex];
      const inputGradients = Array.from({ length: sequenceLength }, () => zeros(d));
      const attendedGradients = Array.from({ length: sequenceLength }, () => zeros(d));
      const queryGradients = Array.from({ length: sequenceLength }, () => zeros(d));
      const keyGradients = Array.from({ length: sequenceLength }, () => zeros(d));
      const valueGradients = Array.from({ length: sequenceLength }, () => zeros(d));

      for (let position = 0; position < sequenceLength; position += 1) {
        const outputGradient = stateGradients[position];
        const hiddenGradient = zeros(h);
        addOuter(layerGradients.feedForwardOut, layerCache.hiddenStates[position], outputGradient);
        for (let index = 0; index < d; index += 1) {
          layerGradients.feedForwardOutBias[index] += outputGradient[index];
        }
        for (let hiddenIndex = 0; hiddenIndex < h; hiddenIndex += 1) {
          let value = 0;
          for (let index = 0; index < d; index += 1) {
            value += layerWeights.feedForwardOut[hiddenIndex][index] * outputGradient[index];
          }
          hiddenGradient[hiddenIndex] = layerCache.hiddenStates[position][hiddenIndex] > 0 ? value : 0;
        }

        const residualGradient = outputGradient.slice();
        addOuter(layerGradients.feedForwardIn, layerCache.residualStates[position], hiddenGradient);
        for (let hiddenIndex = 0; hiddenIndex < h; hiddenIndex += 1) {
          layerGradients.feedForwardInBias[hiddenIndex] += hiddenGradient[hiddenIndex];
          for (let index = 0; index < d; index += 1) {
            residualGradient[index] += layerWeights.feedForwardIn[index][hiddenIndex] * hiddenGradient[hiddenIndex];
          }
        }

        inputGradients[position] = residualGradient.slice();
        addOuter(layerGradients.attentionOutput, layerCache.attentionStates[position], residualGradient);
        for (let inputIndex = 0; inputIndex < d; inputIndex += 1) {
          for (let outputIndex = 0; outputIndex < d; outputIndex += 1) {
            attendedGradients[position][inputIndex] +=
              layerWeights.attentionOutput[inputIndex][outputIndex] * residualGradient[outputIndex];
          }
        }
      }

      for (let position = 0; position < sequenceLength; position += 1) {
        for (let head = 0; head < ATTENTION_HEADS; head += 1) {
          const headStart = head * HEAD_DIMENSION;
          const attention = layerCache.attentions[position][head];
          const scoreGradients = [];
          let weightedScoreGradient = 0;

          for (let source = 0; source <= position; source += 1) {
            for (let index = headStart; index < headStart + HEAD_DIMENSION; index += 1) {
              valueGradients[source][index] += attention[source] * attendedGradients[position][index];
            }
            let gradient = 0;
            for (let index = headStart; index < headStart + HEAD_DIMENSION; index += 1) {
              gradient += attendedGradients[position][index] * layerCache.values[source][index];
            }
            scoreGradients.push(gradient);
            weightedScoreGradient += attention[source] * gradient;
          }

          for (let source = 0; source <= position; source += 1) {
            const scoreGradient = attention[source] *
              (scoreGradients[source] - weightedScoreGradient) / Math.sqrt(HEAD_DIMENSION);
            for (let index = headStart; index < headStart + HEAD_DIMENSION; index += 1) {
              queryGradients[position][index] += scoreGradient * layerCache.keys[source][index];
              keyGradients[source][index] += scoreGradient * layerCache.queries[position][index];
            }
          }
        }
      }

      for (let position = 0; position < sequenceLength; position += 1) {
        addOuter(layerGradients.query, layerCache.states[position], queryGradients[position]);
        addOuter(layerGradients.key, layerCache.states[position], keyGradients[position]);
        addOuter(layerGradients.value, layerCache.states[position], valueGradients[position]);
        for (let inputIndex = 0; inputIndex < d; inputIndex += 1) {
          for (let outputIndex = 0; outputIndex < d; outputIndex += 1) {
            inputGradients[position][inputIndex] +=
              layerWeights.query[inputIndex][outputIndex] * queryGradients[position][outputIndex] +
              layerWeights.key[inputIndex][outputIndex] * keyGradients[position][outputIndex] +
              layerWeights.value[inputIndex][outputIndex] * valueGradients[position][outputIndex];
          }
        }
      }
      stateGradients.splice(0, sequenceLength, ...inputGradients);
    }

    for (let position = 0; position < sequenceLength; position += 1) {
      const tokenGradient = gradients.tokenEmbedding[cache.ids[position]];
      const positionGradient = gradients.positionEmbedding[position];
      for (let index = 0; index < d; index += 1) {
        tokenGradient[index] += stateGradients[position][index];
        positionGradient[index] += stateGradients[position][index];
      }
    }

    let squaredNorm = 0;
    function sumSquares(value) {
      if (Array.isArray(value)) {
        for (const child of value) sumSquares(child);
      } else if (value && typeof value === "object") {
        for (const child of Object.values(value)) sumSquares(child);
      } else {
        squaredNorm += value * value;
      }
    }
    sumSquares(gradients);
    const norm = Math.sqrt(squaredNorm);
    if (!Number.isFinite(norm)) throw new Error("훈련 중 유효하지 않은 기울기가 발생했습니다. 학습 문장이나 훈련 횟수를 줄여 보세요.");
    const clipScale = norm > 1 ? 1 / norm : 1;
    const learningRate = 0.01;

    function updateInPlace(parameter, gradient) {
      if (Array.isArray(parameter)) {
        for (let index = 0; index < parameter.length; index += 1) {
          updateInPlace(parameter[index], gradient[index]);
        }
      } else if (parameter && typeof parameter === "object") {
        for (const key of Object.keys(parameter)) updateInPlace(parameter[key], gradient[key]);
      } else {
        return parameter - learningRate * gradient * clipScale;
      }
      return parameter;
    }
    for (const key of Object.keys(weights)) {
      weights[key] = updateInPlace(weights[key], gradients[key]);
    }
    return loss / targets.length;
  }

  function matchEnding(stem, candidate) {
    if (stem.endsWith(candidate) && stem.length > candidate.length) {
      return { stem: stem.slice(0, -candidate.length), ending: candidate };
    }
    const [initial, ...tailParts] = candidate;
    const tail = tailParts.join("");
    if (!HANGUL_FINAL_CONSONANTS.includes(initial)) return null;
    const prefix = tail ? stem.slice(0, -tail.length) : stem;
    const prefixCharacters = [...prefix];
    const syllable = prefixCharacters[prefixCharacters.length - 1];
    if (!syllable) return null;
    const codePoint = syllable.codePointAt(0);
    const syllableIndex = codePoint - 0xac00;
    if (syllableIndex < 0 || syllableIndex >= 11172) return null;
    const finalIndex = syllableIndex % 28;
    if (!finalIndex || HANGUL_FINAL_CONSONANTS[finalIndex] !== initial) return null;
    const openSyllable = String.fromCodePoint(codePoint - finalIndex);
    const nextStem = `${prefix.slice(0, -syllable.length)}${openSyllable}`;
    if (!nextStem) return null;
    return { stem: nextStem, ending: candidate };
  }

  function tokenizeText(model, text) {
    if (typeof text !== "string") throw new Error("토큰화할 문장은 문자열이어야 합니다.");
    const tokenMap = new Map(model.vocabulary.map((token, index) => [token, index]));
    const segments = [];
    const tokens = [];
    let currentSegment = [];
    let unknownCharacters = 0;

    const appendToken = token => {
      const id = tokenMap.get(token);
      if (id === undefined) {
        unknownCharacters += [...token].length;
        if (currentSegment.length) segments.push(currentSegment);
        currentSegment = [];
      } else {
        currentSegment.push(id);
        tokens.push(token);
      }
    };
    const appendWord = word => {
      if (tokenMap.has(word)) {
        appendToken(word);
        return;
      }
      let stem = word;
      const suffixes = [];
      while (stem) {
        let endingMatch = null;
        for (const candidate of model.endings) {
          const match = tokenMap.has(candidate) ? matchEnding(stem, candidate) : null;
          if (match && (!endingMatch || [...candidate].length > [...endingMatch.ending].length)) {
            endingMatch = match;
          }
        }
        if (!endingMatch) break;
        suffixes.push(endingMatch.ending);
        stem = endingMatch.stem;
        if (tokenMap.has(stem)) break;
      }
      const stemTokens = tokenMap.has(stem) ? [stem] : [...stem];
      const pieces = [...stemTokens, ...suffixes.reverse()];
      for (const piece of pieces) appendToken(piece);
    };

    let lastIndex = 0;
    for (const match of text.matchAll(WORD_PATTERN)) {
      for (const character of text.slice(lastIndex, match.index)) appendToken(character);
      appendWord(match[0]);
      lastIndex = match.index + match[0].length;
    }
    for (const character of text.slice(lastIndex)) appendToken(character);
    if (currentSegment.length) segments.push(currentSegment);
    return { segments, tokens, unknownCharacters };
  }

  function trainStep(model, corpus, contextLength = CONTEXT_LENGTH) {
    if (!validatedModels.has(model)) validateModel(model);
    if (typeof corpus !== "string") throw new Error("훈련 문장은 문자열이어야 합니다.");
    if (!Number.isInteger(contextLength) || contextLength < 1 || contextLength > CONTEXT_LENGTH) {
      throw new Error(`훈련 문맥은 1부터 ${CONTEXT_LENGTH}토큰 사이여야 합니다.`);
    }
    const encoded = tokenizeText(model, corpus);
    const segments = encoded.segments.filter(segment => segment.length > 1);
    if (!segments.length) throw new Error("모델 어휘에 포함된 연속 토큰 두 개 이상을 찾지 못했습니다.");

    const ids = segments[Math.floor(Math.random() * segments.length)];
    const length = Math.min(contextLength, ids.length - 1);
    const start = Math.floor(Math.random() * (ids.length - length));
    const inputs = ids.slice(start, start + length);
    const targets = ids.slice(start + 1, start + length + 1);
    const cache = forward(model, inputs);
    return backpropagate(model, cache, targets);
  }

  function generate(model, prompt, maximumTokens = 40, temperature = 0.8) {
    validateModel(model);
    const encodedPrompt = tokenizeText(model, prompt);
    const result = encodedPrompt.segments.flat();
    if (!result.length) result.push(Math.floor(Math.random() * model.vocabulary.length));

    for (let step = 0; step < maximumTokens; step += 1) {
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
      layers: TRANSFORMER_LAYERS,
      heads: ATTENTION_HEADS,
    }),
    createModel,
    validateModel,
    trainStep,
    generate,
    tokenize: tokenizeText,
  });
})();
