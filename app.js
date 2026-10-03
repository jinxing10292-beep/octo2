const STORAGE_KEYS = {
  examples: "octo-ai-training-examples-v1",
  documents: "octo-ai-local-documents-v1",
};

const BUILT_IN_DOCUMENTS = [
  {
    id: "prototype-overview",
    title: "프로토타입 안내",
    url: "#prototype",
    text: "이 프로토타입은 한국어로 질문을 입력하면 브라우저에 준비된 자료와 사용자가 추가한 자료에서 관련 내용을 검색합니다. 검색된 자료의 문장을 근거로 보여주며, 관련 자료가 없으면 모른다고 답합니다. 현재 생성형 언어 모델은 연결되어 있지 않아 자유로운 문장 생성이나 일반 지식 답변은 제공하지 않습니다.",
  },
  {
    id: "training-status",
    title: "모델 및 훈련 상태",
    url: "#training",
    text: "훈련실은 무작위 초기화한 소형 문자 단위 디코더 트랜스포머를 CPU에서 다음 문자 예측으로 실제 훈련하고 가중치를 JSON으로 내보낼 수 있습니다. 이 미니 모델은 1개 층, 1개 어텐션 헤드, 은닉 차원 16, 문맥 32자 구조이며 일반적인 대화형 언어 모델이나 0.6B 모델이 아닙니다. 0.6B 모델의 훈련·추론에 필요한 자원은 별도로 마련해야 합니다. 훈련 데이터와 모델 가중치는 자동으로 저장되지 않으며 사용자가 내보내 보관해야 합니다.",
  },
  {
    id: "data-and-sources",
    title: "자료 및 출처 사용",
    url: "#sources",
    text: "채팅 화면의 검색 답변은 저장된 자료에서 찾은 내용에만 근거합니다. 자료를 찾지 못하거나 질문과 자료의 관련성이 낮으면 추측하지 않고 모른다고 답합니다. 별도 훈련실의 미니 언어 모델은 자료 검색 답변과 구분된 다음 문자 예측 실험이며, 일반적인 질의응답 능력을 보장하지 않습니다.",
  },
];

const conversation = document.querySelector("#conversation");
const chatForm = document.querySelector("#chat-form");
const chatInput = document.querySelector("#chat-input");
const datasetForm = document.querySelector("#dataset-form");
const promptInput = document.querySelector("#example-prompt");
const responseInput = document.querySelector("#example-response");
const exampleList = document.querySelector("#example-list");
const datasetEmpty = document.querySelector("#dataset-empty");
const toast = document.querySelector("#toast");
let toastTimer;
let transformerModel = null;
let transformerTraining = false;
let cancelTransformerTraining = false;

function readStoredArray(key) {
  try {
    const parsed = JSON.parse(localStorage.getItem(key) || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    console.error(`Could not read ${key} from local storage.`, error);
    return [];
  }
}

function saveArray(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch (error) {
    console.error(`Could not save ${key} to local storage.`, error);
    showToast("브라우저 저장 공간에 저장하지 못했습니다.");
    return false;
  }
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toast.classList.remove("show"), 2800);
}

function createId() {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function makeMessage(role, text, sources = []) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  if (role === "assistant") {
    const avatar = document.createElement("span");
    avatar.className = "assistant-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = "o";
    article.append(avatar);
  }

  const body = document.createElement("div");
  body.className = "message-body";
  const label = document.createElement("p");
  label.className = "message-label";
  label.textContent = role === "assistant" ? "Octo · 검색 기반 답변" : "나";
  const content = document.createElement("p");
  content.className = "message-text";
  content.textContent = text;
  body.append(label, content);

  if (sources.length) {
    const citationList = document.createElement("div");
    citationList.className = "citation-list";
    citationList.setAttribute("aria-label", "답변 출처");
    sources.forEach((source, index) => {
      const citation = document.createElement("a");
      citation.className = "citation";
      citation.href = source.url;
      if (source.url.startsWith("#")) {
        citation.addEventListener("click", event => event.preventDefault());
      } else {
        citation.target = "_blank";
        citation.rel = "noopener noreferrer";
      }
      const number = document.createElement("span");
      number.className = "citation-number";
      number.textContent = `[${index + 1}]`;
      const title = document.createElement("span");
      title.textContent = source.title;
      citation.append(number, title);
      citationList.append(citation);
    });
    body.append(citationList);
  }

  if (role === "assistant") {
    const note = document.createElement("span");
    note.className = "answer-note";
    note.textContent = "검색된 자료를 바탕으로 한 프로토타입 응답";
    body.append(note);
  }

  article.append(body);
  conversation.append(article);
  article.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function normalizeText(text) {
  return text.toLocaleLowerCase("ko-KR").replace(/[^\p{L}\p{N}\s]/gu, " ").replace(/\s+/g, " ").trim();
}

function textTerms(text) {
  const words = normalizeText(text).split(" ").filter(word => word.length > 1);
  const terms = new Set(words);
  for (const word of words) {
    if (word.length > 2) {
      for (let index = 0; index < word.length - 1; index += 1) {
        terms.add(word.slice(index, index + 2));
      }
    }
  }
  return terms;
}

function getDocuments() {
  const customDocuments = readStoredArray(STORAGE_KEYS.documents);
  return [...BUILT_IN_DOCUMENTS, ...customDocuments];
}

function searchDocuments(query) {
  const queryTerms = textTerms(query);
  if (!queryTerms.size) return [];

  return getDocuments()
    .map(document => {
      const content = `${document.title} ${document.text}`;
      const documentTerms = textTerms(content);
      let matched = 0;
      for (const term of queryTerms) {
        if (documentTerms.has(term)) matched += 1;
      }
      const overlap = matched / queryTerms.size;
      const phraseBonus = normalizeText(document.text).includes(normalizeText(query)) ? 0.5 : 0;
      return { document, score: overlap + phraseBonus };
    })
    .filter(result => result.score >= 0.12)
    .sort((left, right) => right.score - left.score)
    .slice(0, 3)
    .map(result => result.document);
}

function answerFromSources(query, sources) {
  if (!sources.length) {
    return {
      text: "현재 저장된 자료에서는 답을 찾지 못했어요. 근거 없이 추측하지 않겠습니다. 관련 문서를 훈련 데이터 탭에서 추가한 뒤 다시 질문해 주세요.",
      sources: [],
    };
  }

  const queryTerms = textTerms(query);
  const excerpts = [];
  const citedSources = [];
  for (const source of sources) {
    const sentences = source.text.split(/(?<=[.!?。])\s+|(?<=니다\.)\s*/u).map(sentence => sentence.trim()).filter(Boolean);
    const bestSentence = sentences
      .map(sentence => {
        const terms = textTerms(sentence);
        let matches = 0;
        for (const term of queryTerms) if (terms.has(term)) matches += 1;
        return { sentence, score: matches / Math.max(queryTerms.size, 1) };
      })
      .sort((left, right) => right.score - left.score)[0];
    if (bestSentence && bestSentence.score > 0) {
      excerpts.push(bestSentence.sentence);
      citedSources.push(source);
    }
  }

  if (!excerpts.length) {
    return {
      text: "관련된 자료는 찾았지만 질문에 답할 만큼 직접적인 근거는 확인하지 못했어요. 더 구체적인 질문이나 관련 문서를 추가해 주세요.",
      sources: [],
    };
  }
  return { text: excerpts.join("\n\n"), sources: citedSources };
}

function submitQuestion(question) {
  const trimmed = question.trim();
  if (!trimmed) return;
  makeMessage("user", trimmed);
  chatInput.value = "";
  chatInput.style.height = "auto";

  const result = answerFromSources(trimmed, searchDocuments(trimmed));
  makeMessage("assistant", result.text, result.sources);
}

chatForm.addEventListener("submit", event => {
  event.preventDefault();
  submitQuestion(chatInput.value);
});

chatInput.addEventListener("input", () => {
  chatInput.style.height = "auto";
  chatInput.style.height = `${Math.min(chatInput.scrollHeight, 150)}px`;
});

chatInput.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

document.querySelectorAll(".suggestion").forEach(button => {
  button.addEventListener("click", () => submitQuestion(button.dataset.prompt || ""));
});

document.querySelector("#clear-chat").addEventListener("click", () => {
  conversation.replaceChildren();
  showToast("대화를 지웠습니다.");
});

document.querySelectorAll(".nav-item").forEach(button => {
  button.addEventListener("click", () => {
    const selectedTab = button.dataset.tab;
    const isChat = selectedTab === "chat";
    document.querySelectorAll(".nav-item").forEach(navItem => {
      navItem.classList.toggle("active", navItem === button);
      if (navItem === button) navItem.setAttribute("aria-current", "page");
      else navItem.removeAttribute("aria-current");
    });
    document.querySelector("#chat-page").hidden = !isChat;
    document.querySelector("#training-page").hidden = isChat;
    document.querySelector("#page-title").textContent = isChat ? "대화" : "훈련 데이터";
    document.querySelector("#clear-chat").hidden = !isChat;
    if (!isChat) renderExamples();
  });
});

function renderExamples() {
  const examples = readStoredArray(STORAGE_KEYS.examples);
  exampleList.replaceChildren();
  document.querySelector("#dataset-count").textContent = `${examples.length}개`;
  datasetEmpty.hidden = examples.length > 0;
  examples.forEach((example, index) => {
    const card = document.createElement("article");
    card.className = "example-card";
    const copy = document.createElement("div");
    copy.className = "example-copy";
    const prompt = document.createElement("strong");
    prompt.textContent = `사용자: ${example.prompt}`;
    const response = document.createElement("p");
    response.textContent = `답변: ${example.response}`;
    copy.append(prompt, response);
    const remove = document.createElement("button");
    remove.className = "delete-example";
    remove.type = "button";
    remove.textContent = "삭제";
    remove.setAttribute("aria-label", `예시 ${index + 1} 삭제`);
    remove.addEventListener("click", () => {
      const current = readStoredArray(STORAGE_KEYS.examples);
      current.splice(index, 1);
      if (saveArray(STORAGE_KEYS.examples, current)) renderExamples();
    });
    card.append(copy, remove);
    exampleList.append(card);
  });
}

function renderDocuments() {
  const documents = readStoredArray(STORAGE_KEYS.documents);
  const list = document.querySelector("#document-list");
  list.replaceChildren();
  document.querySelector("#document-count").textContent = `내 자료 ${documents.length}개`;
  documents.forEach((source, index) => {
    const row = document.createElement("div");
    row.className = "document-row";
    const link = document.createElement("a");
    link.textContent = source.title;
    link.href = source.url || `#${source.id}`;
    if (source.url) {
      link.target = "_blank";
      link.rel = "noopener noreferrer";
    } else {
      link.addEventListener("click", event => event.preventDefault());
    }
    const preview = document.createElement("span");
    preview.textContent = source.text.length > 90 ? `${source.text.slice(0, 90)}…` : source.text;
    link.append(preview);
    const remove = document.createElement("button");
    remove.className = "delete-example";
    remove.type = "button";
    remove.textContent = "삭제";
    remove.setAttribute("aria-label", `${source.title} 자료 삭제`);
    remove.addEventListener("click", () => {
      const current = readStoredArray(STORAGE_KEYS.documents);
      current.splice(index, 1);
      if (saveArray(STORAGE_KEYS.documents, current)) {
        renderDocuments();
        showToast("검색 자료를 삭제했습니다.");
      }
    });
    row.append(link, remove);
    list.append(row);
  });
}

document.querySelector("#document-form").addEventListener("submit", event => {
  event.preventDefault();
  const title = document.querySelector("#document-title").value.trim();
  const urlValue = document.querySelector("#document-url").value.trim();
  const text = document.querySelector("#document-content").value.trim();
  if (!title || !text) return;

  let url = "";
  if (urlValue) {
    try {
      const parsed = new URL(urlValue);
      if (parsed.protocol !== "https:" && parsed.protocol !== "http:") throw new Error("Only HTTP(S) source links are allowed.");
      url = parsed.href;
    } catch (error) {
      showToast("출처 링크는 http 또는 https 주소여야 합니다.");
      return;
    }
  }

  const documents = readStoredArray(STORAGE_KEYS.documents);
  documents.unshift({ id: `user-${createId()}`, title, url, text });
  if (saveArray(STORAGE_KEYS.documents, documents)) {
    event.currentTarget.reset();
    renderDocuments();
    showToast("검색 자료를 추가했습니다.");
  }
});

datasetForm.addEventListener("submit", event => {
  event.preventDefault();
  const prompt = promptInput.value.trim();
  const response = responseInput.value.trim();
  if (!prompt || !response) return;
  const examples = readStoredArray(STORAGE_KEYS.examples);
  examples.unshift({ prompt, response, createdAt: new Date().toISOString() });
  if (saveArray(STORAGE_KEYS.examples, examples)) {
    datasetForm.reset();
    renderExamples();
    showToast("한국어 대화 예시를 저장했습니다.");
  }
});

document.querySelector("#export-data").addEventListener("click", () => {
  const examples = readStoredArray(STORAGE_KEYS.examples);
  const blob = new Blob([JSON.stringify({ format: "octo-ai-dataset-v1", language: "ko", examples }, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "octo-ai-training-data.json";
  link.click();
  URL.revokeObjectURL(url);
  showToast(`${examples.length}개 예시를 JSON으로 내보냈습니다.`);
});

document.querySelector("#import-data").addEventListener("click", () => document.querySelector("#import-file").click());
document.querySelector("#import-file").addEventListener("change", async event => {
  const [file] = event.target.files || [];
  if (!file) return;
  try {
    const imported = JSON.parse(await file.text());
    const examples = Array.isArray(imported) ? imported : imported.examples;
    if (!Array.isArray(examples) || examples.some(example => typeof example.prompt !== "string" || typeof example.response !== "string")) {
      throw new Error("Expected an array of prompt/response examples.");
    }
    const cleanExamples = examples.map(example => ({
      prompt: example.prompt.trim().slice(0, 4000),
      response: example.response.trim().slice(0, 8000),
      createdAt: typeof example.createdAt === "string" ? example.createdAt : new Date().toISOString(),
    })).filter(example => example.prompt && example.response);
    if (saveArray(STORAGE_KEYS.examples, [...cleanExamples, ...readStoredArray(STORAGE_KEYS.examples)])) {
      renderExamples();
      showToast(`${cleanExamples.length}개 예시를 가져왔습니다.`);
    }
  } catch (error) {
    console.error("Could not import training examples.", error);
    showToast("JSON 형식이 올바르지 않습니다.");
  } finally {
    event.target.value = "";
  }
});

function setTransformerModel(model, status = "모델 준비됨") {
  transformerModel = window.OCTO_MINI_GPT.validateModel(model);
  document.querySelector("#transformer-state").textContent =
    `${status} · 문자 ${transformerModel.vocabulary.length}종`;
  document.querySelector("#generate-transformer-sample").disabled = false;
  document.querySelector("#export-transformer-weights").disabled = false;
  document.querySelector("#copy-transformer-weights").disabled = false;
}

function setTransformerTrainingControls(isRunning) {
  transformerTraining = isRunning;
  document.querySelector("#start-transformer-training").disabled = isRunning;
  document.querySelector("#initialize-transformer").disabled = isRunning;
  document.querySelector("#use-examples-as-corpus").disabled = isRunning;
  document.querySelector("#training-steps").disabled = isRunning;
  document.querySelector("#stop-transformer-training").disabled = !isRunning;
}

document.querySelector("#use-examples-as-corpus").addEventListener("click", () => {
  const examples = readStoredArray(STORAGE_KEYS.examples);
  if (!examples.length) {
    showToast("먼저 대화 예시를 추가하세요.");
    return;
  }
  document.querySelector("#training-corpus").value = examples
    .map(example => `사용자: ${example.prompt}\n답변: ${example.response}`)
    .join("\n\n");
  showToast(`${examples.length}개 대화 예시를 학습 문장에 넣었습니다.`);
});

function readTrainingCorpus() {
  const corpus = document.querySelector("#training-corpus").value.trim();
  if ([...corpus].length < 2) throw new Error("훈련 문장을 두 글자 이상 입력하세요.");
  return corpus;
}

document.querySelector("#initialize-transformer").addEventListener("click", () => {
  try {
    const model = window.OCTO_MINI_GPT.createModel(readTrainingCorpus());
    setTransformerModel(model, "무작위 초기화");
    document.querySelector("#training-progress-fill").style.width = "0%";
    document.querySelector("#training-status").textContent = "무작위 가중치 초기화 완료. 훈련 전에는 생성 결과가 무작위입니다.";
    document.querySelector("#training-loss").textContent = "";
    document.querySelector("#generation-output").textContent = "아직 훈련되지 않은 무작위 출력입니다.";
  } catch (error) {
    console.error("Could not initialize the mini transformer.", error);
    showToast(error.message || "모델 초기화에 실패했습니다.");
  }
});

document.querySelector("#start-transformer-training").addEventListener("click", () => {
  let corpus;
  let steps;
  let vocabularyCoverage = 0;
  try {
    corpus = readTrainingCorpus();
    steps = Number.parseInt(document.querySelector("#training-steps").value, 10);
    if (!Number.isInteger(steps) || steps < 1 || steps > 3000) {
      throw new Error("훈련 단계는 1부터 3000 사이의 정수여야 합니다.");
    }
    if (!transformerModel) {
      transformerModel = window.OCTO_MINI_GPT.createModel(corpus);
      setTransformerModel(transformerModel, "무작위 초기화");
    } else {
      window.OCTO_MINI_GPT.validateModel(transformerModel);
    }
    const modelVocabulary = new Set(transformerModel.vocabulary);
    const corpusCharacters = [...corpus];
    const knownCharacterCount = corpusCharacters.filter(character => modelVocabulary.has(character)).length;
    const hasTrainablePair = corpusCharacters.some((character, index) =>
      modelVocabulary.has(character) && modelVocabulary.has(corpusCharacters[index + 1]),
    );
    if (!hasTrainablePair) {
      throw new Error("현재 모델 어휘와 겹치는 문자가 없습니다. 현재 문장으로 모델을 초기화하세요.");
    }
    vocabularyCoverage = Math.round((knownCharacterCount / corpusCharacters.length) * 100);
  } catch (error) {
    console.error("Could not start mini-transformer training.", error);
    showToast(error.message || "훈련을 시작하지 못했습니다.");
    return;
  }

  cancelTransformerTraining = false;
  setTransformerTrainingControls(true);
  document.querySelector("#training-progress-fill").style.width = "0%";
  document.querySelector("#training-status").textContent = "다음 문자 예측으로 가중치를 갱신하는 중...";
  document.querySelector("#training-loss").textContent = "";
  const losses = [];
  let step = 0;

  const runStep = () => {
    if (cancelTransformerTraining) {
      setTransformerTrainingControls(false);
      document.querySelector("#training-status").textContent = `훈련 중지 · ${step}/${steps}단계 완료`;
      setTransformerModel(transformerModel, "부분 훈련됨");
      return;
    }
    try {
      losses.push(window.OCTO_MINI_GPT.trainStep(transformerModel, corpus));
      step += 1;
      const start = Math.max(0, losses.length - 25);
      const recentLoss = losses.slice(start).reduce((sum, loss) => sum + loss, 0) / (losses.length - start);
      document.querySelector("#training-progress-fill").style.width = `${(step / steps) * 100}%`;
      document.querySelector("#training-status").textContent = `훈련 중 · ${step}/${steps}단계 · 어휘 적용률 ${vocabularyCoverage}%`;
      document.querySelector("#training-loss").textContent = `최근 손실 ${recentLoss.toFixed(3)}`;
      if (step < steps) {
        window.setTimeout(runStep, 0);
      } else {
        setTransformerTrainingControls(false);
        setTransformerModel(transformerModel, "훈련 완료");
        document.querySelector("#training-status").textContent = `훈련 완료 · ${steps}단계 · 학습률 0.08`;
        showToast("미니 트랜스포머 훈련을 마쳤습니다. 가중치를 JSON으로 내보내 보관하세요.");
      }
    } catch (error) {
      console.error("Mini-transformer training failed.", error);
      setTransformerTrainingControls(false);
      document.querySelector("#training-status").textContent = "훈련에 실패했습니다.";
      showToast(error.message || "훈련 도중 오류가 발생했습니다.");
    }
  };
  window.setTimeout(runStep, 0);
});

document.querySelector("#stop-transformer-training").addEventListener("click", () => {
  cancelTransformerTraining = true;
  document.querySelector("#training-status").textContent = "현재 단계를 마친 뒤 중지합니다...";
});

document.querySelector("#generate-transformer-sample").addEventListener("click", () => {
  if (!transformerModel) return;
  try {
    const prompt = document.querySelector("#generation-prompt").value;
    const generated = window.OCTO_MINI_GPT.generate(transformerModel, prompt, 160, 0.8);
    const modelVocabulary = new Set(transformerModel.vocabulary);
    const ignoredPromptCharacters = [...prompt].filter(character => !modelVocabulary.has(character)).length;
    document.querySelector("#generation-output").textContent = ignoredPromptCharacters
      ? `${generated}\n\n(모델 어휘에 없는 시작 문자는 ${ignoredPromptCharacters}개 제외했습니다.)`
      : generated;
  } catch (error) {
    console.error("Could not generate a mini-transformer sample.", error);
    showToast(error.message || "문장 생성에 실패했습니다.");
  }
});

function transformerWeightsJSON() {
  if (!transformerModel) throw new Error("먼저 모델을 초기화하거나 훈련하세요.");
  return JSON.stringify(transformerModel);
}

document.querySelector("#export-transformer-weights").addEventListener("click", () => {
  try {
    const json = JSON.stringify(transformerModel, null, 2);
    document.querySelector("#transformer-weights").value = json;
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "octo-mini-decoder-weights.json";
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    showToast("미니 트랜스포머 가중치를 JSON 파일로 내보냈습니다.");
  } catch (error) {
    console.error("Could not export mini-transformer weights.", error);
    showToast(error.message || "가중치 내보내기에 실패했습니다.");
  }
});

document.querySelector("#copy-transformer-weights").addEventListener("click", async () => {
  try {
    const json = transformerWeightsJSON();
    document.querySelector("#transformer-weights").value = JSON.stringify(transformerModel, null, 2);
    if (!navigator.clipboard || typeof navigator.clipboard.writeText !== "function") {
      document.querySelector("#transformer-weights").focus();
      document.querySelector("#transformer-weights").select();
      throw new Error("브라우저 클립보드 권한을 사용할 수 없습니다. 선택된 JSON을 직접 복사하세요.");
    }
    await navigator.clipboard.writeText(json);
    showToast("가중치 JSON을 클립보드에 복사했습니다.");
  } catch (error) {
    console.error("Could not copy mini-transformer weights.", error);
    showToast(error.message || "가중치 복사에 실패했습니다.");
  }
});

document.querySelector("#import-transformer-weights").addEventListener("click", () => {
  try {
    const imported = JSON.parse(document.querySelector("#transformer-weights").value);
    setTransformerModel(imported, "JSON 불러옴");
    document.querySelector("#training-status").textContent = "가중치를 불러왔습니다. 같은 어휘에 포함된 문장으로 이어서 훈련하거나 출력을 생성할 수 있습니다.";
    document.querySelector("#generation-output").textContent = "가중치를 불러왔습니다. 시작 문구를 입력하고 생성을 눌러 보세요.";
    showToast("가중치 JSON을 검증하고 불러왔습니다.");
  } catch (error) {
    console.error("Could not import mini-transformer weights.", error);
    showToast(error.message || "가중치 JSON을 불러오지 못했습니다.");
  }
});

function checkBrowserReadiness() {
  const supportsWebGPU = Boolean(navigator.gpu);
  document.querySelector("#webgpu-status").textContent = supportsWebGPU
    ? "이 브라우저에서 API를 사용할 수 있습니다"
    : "이 브라우저에서 지원하지 않습니다";
  document.querySelector("#webgpu-tag").textContent = supportsWebGPU ? "지원" : "미지원";
  document.querySelector("#webgpu-tag").classList.toggle("unsupported", !supportsWebGPU);

  const memory = navigator.deviceMemory;
  if (typeof memory === "number") {
    document.querySelector("#memory-status").textContent = `브라우저 보고값 약 ${memory} GB (시스템 전체 RAM과 다를 수 있음)`;
    document.querySelector("#memory-tag").textContent = "참고";
  } else {
    document.querySelector("#memory-status").textContent = "브라우저에서 메모리 정보를 제공하지 않습니다";
    document.querySelector("#memory-tag").textContent = "비공개";
  }
}

function normalizeVocabulary() {
  const source = window.OCTO_KO_VOCABULARY;
  const normalized = {};
  for (const [category, definition] of Object.entries(source)) {
    const entries = [];
    if (definition.groups) {
      for (const [group, words] of Object.entries(definition.groups)) {
        for (const word of words.trim().split(/\s+/u)) {
          entries.push({ word, group });
        }
      }
    } else {
      const wordList = [definition.items, definition.additional || "", definition.more || ""].filter(Boolean).join(" ");
      for (const word of wordList.trim().split(/\s+/u)) {
        entries.push({ word, group: "" });
      }
    }
    const seen = new Set();
    const uniqueEntries = entries.filter(entry => {
      if ((category === "verbs" || category === "adjectives") && !entry.word.endsWith("-")) return false;
      if (seen.has(entry.word)) return false;
      seen.add(entry.word);
      return true;
    });
    normalized[category] = {
      label: definition.label,
      entries: uniqueEntries.slice(0, 500),
    };
  }
  return normalized;
}

const koreanVocabulary = normalizeVocabulary();
const vocabularyCategory = document.querySelector("#vocabulary-category");
const vocabularySearch = document.querySelector("#vocabulary-search");
const vocabularyList = document.querySelector("#vocabulary-list");

function renderVocabulary() {
  const category = koreanVocabulary[vocabularyCategory.value];
  const query = vocabularySearch.value.trim().toLocaleLowerCase("ko-KR");
  const entries = category.entries.filter(entry => !query || entry.word.includes(query) || entry.group.includes(query));
  vocabularyList.replaceChildren();
  document.querySelector("#vocabulary-count").textContent = `${entries.length}개 / 전체 ${category.entries.length}개`;
  if (!entries.length) {
    const empty = document.createElement("p");
    empty.className = "vocabulary-empty";
    empty.textContent = "검색 결과가 없습니다.";
    vocabularyList.append(empty);
    return;
  }
  for (const entry of entries) {
    const chip = document.createElement("span");
    chip.className = "vocabulary-chip";
    chip.textContent = entry.word;
    if (entry.group) {
      chip.title = entry.group;
      chip.dataset.group = entry.group;
    }
    vocabularyList.append(chip);
  }
}

vocabularyCategory.addEventListener("change", renderVocabulary);
vocabularySearch.addEventListener("input", renderVocabulary);
document.querySelector("#export-vocabulary").addEventListener("click", () => {
  const data = Object.fromEntries(
    Object.entries(koreanVocabulary).map(([key, category]) => [
      key,
      { label: category.label, entries: category.entries },
    ]),
  );
  const blob = new Blob([JSON.stringify({ format: "octo-ai-korean-vocabulary-v1", language: "ko", categories: data }, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "octo-ai-korean-vocabulary.json";
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  showToast("품사별 한국어 어휘를 JSON으로 내보냈습니다.");
});

renderExamples();
renderDocuments();
checkBrowserReadiness();
renderVocabulary();
makeMessage("assistant", "안녕하세요! 저는 지금 저장된 자료를 검색해 근거와 함께 답하는 초기 프로토타입이에요. 아직 생성형 AI 모델은 연결되어 있지 않으니, 답을 자료에서 찾지 못하면 모른다고 말씀드릴게요.");
