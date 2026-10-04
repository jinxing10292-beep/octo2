const STORAGE_KEYS = {
  examples: "octo-ai-training-examples-v1",
  documents: "octo-ai-local-documents-v1",
};
const MODEL_DATABASE = "octo-ai-model-storage";
const MODEL_STORE = "checkpoints";
const MODEL_STORAGE_KEY = "latest-ko-word-ending-char-4l-4h-256d-512c-2048v-int8";
const TRAINING_CHECKPOINT_INTERVAL = 25;
const WEIGHTS_PACKAGE_FORMAT = "octo-mini-weights-package-v4-int8";
const MAX_WEIGHTS_FILE_BYTES = 256 * 1024 * 1024;
const MAX_MODEL_VOCABULARY_SIZE = 2048;
const RESERVED_TRAINING_WORD_TOKENS = 512;
const DEFAULT_VOCABULARY_TARGET = MAX_MODEL_VOCABULARY_SIZE - RESERVED_TRAINING_WORD_TOKENS;

const BUILT_IN_DOCUMENTS = [
  {
    id: "prototype-overview",
    title: "프로토타입 안내",
    url: "#prototype",
    text: "채팅은 저장된 자료에서 근거를 찾아 출처와 함께 답하거나, 자료와 관련이 없는 일상 대화에는 브라우저에 불러온 한국어 미니 언어 모델을 사용합니다. 대화 모델은 규모가 작아 답변이 부정확할 수 있습니다.",
  },
  {
    id: "training-status",
    title: "모델 및 훈련 상태",
    url: "#training",
    text: "훈련실은 단어 전체형을 토큰으로 사용하는 한국어 디코더 트랜스포머를 CPU에서 다음 토큰 예측으로 훈련합니다. 모르는 단어도 문자 조각으로 나누지 않고 남은 어휘 공간에 단어 전체를 추가합니다. 모델은 4개 층, 4개 어텐션 헤드, 차원 256, 피드포워드 256, 문맥 512토큰, 최대 어휘 2,048개이며 int8 양자화로 약 240만 매개변수를 75% 압축하여 저장합니다.",
  },
  {
    id: "data-and-sources",
    title: "자료 및 출처 사용",
    url: "#sources",
    text: "자동 모드에서 저장된 자료와 질문이 직접 일치하면 검색된 문장을 근거로 답하고 출처를 표시합니다. 직접적인 근거가 없으면 브라우저의 미니 언어 모델로 일상 대화를 생성합니다. 자료 검색 모드에서는 근거가 없을 때 모른다고 답합니다. 미니 모델은 작아 일반 지식이나 정확한 답변을 보장하지 않습니다.",
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
let transformerReady = Promise.resolve();
let cancelTransformerTraining = false;
let weightsFileHandle = null;
const chatHistory = [];
const sendChatButton = document.querySelector("#chat-form button[type='submit']");

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

function makeMessage(role, text, sources = [], answerType = sources.length ? "search" : "conversation") {
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
  label.textContent = role !== "assistant"
    ? "나"
    : answerType === "search"
      ? "Octo · 자료 검색"
      : answerType === "conversation"
        ? "Octo · 로컬 미니 모델"
        : "Octo";
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
    note.textContent = answerType === "search"
      ? "저장된 자료에서 찾은 근거"
      : answerType === "conversation"
        ? "작은 로컬 모델의 생성 답변으로 틀릴 수 있습니다"
        : "";
    note.hidden = !note.textContent;
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

function hasDirectSearchMatch(query, sources) {
  const exactTerms = normalizeText(query).split(" ").filter(term => term.length > 1);
  return exactTerms.some(term => sources.some(source =>
    textTerms(`${source.title} ${source.text}`).has(term),
  ));
}

function createConversationPrompt(question) {
  const previousTurns = chatHistory.slice(-6).map(turn =>
    `사용자: ${turn.question}\n답변: ${turn.answer}`,
  );
  return [...previousTurns, `사용자: ${question}\n답변:`].join("\n");
}

async function submitQuestion(question) {
  const trimmed = question.trim();
  if (!trimmed || sendChatButton.disabled) return;
  makeMessage("user", trimmed);
  chatInput.value = "";
  chatInput.style.height = "auto";
  sendChatButton.disabled = true;

  try {
    const mode = document.querySelector("#chat-mode").value;
    const sources = searchDocuments(trimmed);
    if (mode === "search" || (mode === "auto" && hasDirectSearchMatch(trimmed, sources))) {
      const result = answerFromSources(trimmed, sources);
      makeMessage("assistant", result.text, result.sources, "search");
      chatHistory.push({ question: trimmed, answer: result.text });
      return;
    }

    await transformerReady;
    if (!transformerModel) {
      const message = "일상 대화를 생성할 미니 모델을 불러오지 못했습니다. 가중치 로드 상태를 확인하거나 자료 검색 모드로 질문해 주세요.";
      makeMessage("assistant", message, [], "system");
      chatHistory.push({ question: trimmed, answer: message });
      return;
    }

    if (!window.OCTO_MINI_GPT.tokenize(transformerModel, trimmed).segments.flat().length) {
      throw new Error("질문에 모델 어휘로 처리할 수 있는 단어가 없습니다. 자료 검색 모드로 바꾸거나 모델 어휘에 맞는 표현을 사용해 주세요.");
    }
    await new Promise(resolve => window.setTimeout(resolve, 0));
    const response = window.OCTO_MINI_GPT.generateContinuation(
      transformerModel,
      createConversationPrompt(trimmed),
      24,
      0.65,
    ).trim();
    if (!response) throw new Error("모델이 답변 토큰을 생성하지 못했습니다.");
    makeMessage("assistant", response, [], "conversation");
    chatHistory.push({ question: trimmed, answer: response });
  } catch (error) {
    console.error("Could not answer the chat message.", error);
    const message = `답변을 만들지 못했습니다: ${error.message || "알 수 없는 오류"}`;
    makeMessage("assistant", message, [], "system");
    chatHistory.push({ question: trimmed, answer: message });
  } finally {
    sendChatButton.disabled = false;
  }
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
  chatHistory.length = 0;
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
  const existingTokens = new Set(transformerModel.vocabulary);
  const missingSeedTokens = [...new Set(collectVocabularySeedTokens(transformerModel))]
    .filter(token => !existingTokens.has(token))
    .slice(0, Math.max(0, DEFAULT_VOCABULARY_TARGET - transformerModel.vocabulary.length));
  const addedTokens = window.OCTO_MINI_GPT.expandVocabulary(
    transformerModel,
    missingSeedTokens,
  );
  const parameterCount = countParameters(transformerModel.weights);
  document.querySelector("#transformer-state").textContent =
    `${status} · 토큰 ${transformerModel.vocabulary.length}종 · ${parameterCount.toLocaleString("ko-KR")}개 매개변수` +
    (addedTokens ? ` · 기본 토큰 ${addedTokens}종 보충(기존 문자 가중치로 초기화, 신규 문자는 무작위, 추가 학습 필요)` : "");
  document.querySelector("#generate-transformer-sample").disabled = false;
  document.querySelector("#export-transformer-weights").disabled = false;
  document.querySelector("#copy-transformer-weights").disabled = false;
  return addedTokens;
}

function collectVocabularySeedTokens(model) {
  const vocabulary = window.OCTO_KO_VOCABULARY;
  const wordTokens = [];
  const seenWords = new Set();
  for (const [category, definition] of Object.entries(vocabulary)) {
    if (category === "endings") continue;
    const sources = [definition.daily || "", definition.items || "", definition.additional || "", definition.more || ""];
    if (definition.groups) sources.push(...Object.values(definition.groups));
    for (const source of sources) {
      for (let word of source.trim().split(/\s+/u)) {
        if (category === "verbs" || category === "adjectives") word = word.replace(/-$/u, "");
        if (word && !seenWords.has(word)) {
          seenWords.add(word);
          wordTokens.push(word);
        }
      }
    }
  }
  const characterTokens = [...new Set([...wordTokens.join(""), ..." \n\t.,!?;:()[]{}\"'"])];
  const endings = model.endings || [];
  return [...characterTokens, ...endings, ...wordTokens];
}

function countParameters(value) {
  if (Array.isArray(value)) return value.reduce((sum, child) => sum + countParameters(child), 0);
  if (value && typeof value === "object") {
    return Object.values(value).reduce((sum, child) => sum + countParameters(child), 0);
  }
  return 1;
}

function setTransformerTrainingControls(isRunning) {
  document.querySelector("#start-transformer-training").disabled = isRunning;
  document.querySelector("#initialize-transformer").disabled = isRunning;
  document.querySelector("#connect-weights-file").disabled = isRunning;
  document.querySelector("#use-examples-as-corpus").disabled = isRunning;
  document.querySelector("#training-steps").disabled = isRunning;
  document.querySelector("#stop-transformer-training").disabled = !isRunning;
}

function openModelDatabase() {
  if (!("indexedDB" in window)) return Promise.reject(new Error("이 브라우저는 IndexedDB 모델 저장을 지원하지 않습니다."));
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(MODEL_DATABASE, 1);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(MODEL_STORE)) {
        request.result.createObjectStore(MODEL_STORE);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("모델 저장소를 열지 못했습니다."));
  });
}

async function writeModelCheckpoint(packageJSON) {
  const database = await openModelDatabase();
  try {
    await new Promise((resolve, reject) => {
      const transaction = database.transaction(MODEL_STORE, "readwrite");
      transaction.objectStore(MODEL_STORE).put(packageJSON, MODEL_STORAGE_KEY);
      transaction.oncomplete = resolve;
      transaction.onerror = () => reject(transaction.error || new Error("모델 체크포인트를 저장하지 못했습니다."));
      transaction.onabort = () => reject(transaction.error || new Error("모델 체크포인트 저장이 중단되었습니다."));
    });
  } finally {
    database.close();
  }
}

async function readModelCheckpoint() {
  const database = await openModelDatabase();
  try {
    return await new Promise((resolve, reject) => {
      const transaction = database.transaction(MODEL_STORE, "readonly");
      const request = transaction.objectStore(MODEL_STORE).get(MODEL_STORAGE_KEY);
      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(request.error || new Error("저장된 모델 체크포인트를 읽지 못했습니다."));
    });
  } finally {
    database.close();
  }
}

function setWeightsSaveStatus(message) {
  document.querySelector("#weights-save-status").textContent = message;
}

async function saveTrainingCheckpoint(writeConnectedFile = false) {
  if (!transformerModel) return false;
  try {
    const packageData = await makeWeightsPackage();
    const packageJSON = JSON.stringify(packageData);
    if (new TextEncoder().encode(packageJSON).byteLength > MAX_WEIGHTS_FILE_BYTES) {
      throw new Error("가중치 패키지가 256 MiB 제한을 넘어 자동 저장하지 못했습니다.");
    }
    await writeModelCheckpoint(packageJSON);
    if (writeConnectedFile && weightsFileHandle) {
      const permission = await weightsFileHandle.queryPermission({ mode: "readwrite" });
      if (permission !== "granted") {
        throw new Error("연결한 가중치 파일의 쓰기 권한이 만료됐습니다. 파일을 다시 연결하세요.");
      }
      const writable = await weightsFileHandle.createWritable();
      try {
        await writable.write(`${JSON.stringify(packageData, null, 2)}\n`);
        await writable.close();
      } catch (error) {
        await writable.abort();
        throw error;
      }
      setWeightsSaveStatus("이 브라우저의 IndexedDB와 연결한 weights.json 파일에 저장했습니다.");
    } else {
      setWeightsSaveStatus("이 브라우저의 IndexedDB에 체크포인트를 저장했습니다.");
    }
    return true;
  } catch (error) {
    console.error("Could not save mini-transformer checkpoint.", error);
    setWeightsSaveStatus(`자동 저장 오류: ${error.message || "체크포인트를 저장하지 못했습니다."}`);
    return false;
  }
}

async function initializeSavedTransformer() {
  const status = document.querySelector("#training-status");
  status.textContent = "저장된 모델 가중치를 확인하는 중...";
  try {
    const savedPackage = await readModelCheckpoint();
    if (savedPackage) {
      const model = await readWeightsPackage(savedPackage);
      const addedTokens = setTransformerModel(model, "자동 저장 모델");
      if (addedTokens) await saveTrainingCheckpoint(false);
      status.textContent = addedTokens
        ? `${addedTokens}개 기본 토큰을 보충했습니다. 기존 학습값은 보존했으며 새 토큰은 기존 문자 가중치로 초기화하고, 없는 문자는 무작위 초기화해 추가 학습이 필요합니다.`
        : "이 브라우저에 저장된 가중치를 불러왔습니다. 이어서 훈련할 수 있습니다.";
      if (!addedTokens) {
        setWeightsSaveStatus("이 브라우저에 저장된 체크포인트를 불러왔습니다. 이어서 훈련하면 기존 가중치가 갱신됩니다.");
      }
      setTransformerTrainingControls(false);
      return;
    }
  } catch (error) {
    console.error("Could not restore the local model checkpoint.", error);
    setWeightsSaveStatus(`저장된 체크포인트를 불러오지 못했습니다: ${error.message || "저장소 오류"}`);
  }

  try {
    const response = await fetch("./weights.json.gz", { cache: "no-cache" });
    if (!response.ok) throw new Error(`압축 가중치 파일을 읽지 못했습니다 (HTTP ${response.status}).`);
    const contentLength = Number(response.headers.get("content-length"));
    if (Number.isFinite(contentLength) && contentLength > MAX_WEIGHTS_FILE_BYTES) {
      throw new Error("배포된 weights.json.gz가 256 MiB 제한을 넘었습니다.");
    }
    const responseBytes = await response.arrayBuffer();
    const bytes = new Uint8Array(responseBytes);
    const isGzip = bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
    let json;
    if (isGzip) {
      if (typeof DecompressionStream !== "function") {
        throw new Error("이 브라우저는 gzip 가중치 압축 해제를 지원하지 않습니다.");
      }
      const decompressed = new Blob([responseBytes]).stream().pipeThrough(new DecompressionStream("gzip"));
      json = await new Response(decompressed).text();
    } else {
      json = new TextDecoder().decode(responseBytes);
    }
    const model = await readWeightsPackage(json);
    const addedTokens = setTransformerModel(model, "압축 가중치 불러옴");
    const checkpointSaved = addedTokens ? await saveTrainingCheckpoint(false) : false;
    status.textContent = addedTokens
      ? `${addedTokens}개 기본 토큰을 보충했습니다. 기존 학습값은 보존했으며 새 토큰은 기존 문자 가중치로 초기화하고, 없는 문자는 무작위 초기화해 관련 문장으로 학습해야 합니다.`
      : "배포된 압축 가중치를 불러왔습니다. 훈련을 시작하면 기존 가중치에서 이어 학습합니다.";
    if (!addedTokens) {
      setWeightsSaveStatus("배포된 압축 가중치를 불러왔습니다. 훈련 체크포인트는 이 브라우저에 자동 저장됩니다.");
    } else if (checkpointSaved) {
      setWeightsSaveStatus("기존 학습 가중치를 보존하고 기본 어휘를 추가한 체크포인트를 저장했습니다. 새 토큰은 아직 학습되지 않았으므로 훈련 후 가중치 JSON을 내려받으세요.");
    }
  } catch (error) {
    console.error("Could not load bundled compressed weights.", error);
    status.textContent = "기존 가중치를 불러오지 못했습니다. JSON 파일을 선택하거나 무작위 모델을 초기화하세요.";
    setWeightsSaveStatus(`압축 가중치 자동 불러오기 실패: ${error.message || "모델을 검증하지 못했습니다."}`);
  }
  setTransformerTrainingControls(false);
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
  let addedTrainingWords = 0;
  let omittedTrainingWords = 0;
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
    const wordExpansion = window.OCTO_MINI_GPT.addTextWords(transformerModel, corpus);
    addedTrainingWords = wordExpansion.added;
    omittedTrainingWords = wordExpansion.missing.length;
    if (addedTrainingWords) setTransformerModel(transformerModel, "학습 단어 추가");
    const tokenization = window.OCTO_MINI_GPT.tokenize(transformerModel, corpus);
    const corpusCharacters = [...corpus];
    const knownCharacterCount = corpusCharacters.length - tokenization.unknownCharacters;
    if (!tokenization.segments.some(segment => segment.length > 1)) {
      throw new Error("현재 모델 어휘에서 이어지는 훈련 토큰을 찾지 못했습니다. 새 단어나 문자를 추가하거나 현재 문장으로 모델을 초기화하세요.");
    }
    vocabularyCoverage = Math.round((knownCharacterCount / corpusCharacters.length) * 100);
    if (omittedTrainingWords) {
      showToast(`어휘 상한에 도달해 새로운 단어 ${omittedTrainingWords}개를 통째로 추가하지 못했습니다. 기존 단어는 글자로 쪼개지지 않습니다.`);
    }
  } catch (error) {
    console.error("Could not start mini-transformer training.", error);
    showToast(error.message || "훈련을 시작하지 못했습니다.");
    return;
  }

  cancelTransformerTraining = false;
  setTransformerTrainingControls(true);
  document.querySelector("#training-progress-fill").style.width = "0%";
  document.querySelector("#training-status").textContent =
    `다음 토큰 예측으로 가중치를 갱신하는 중...${addedTrainingWords ? ` 새 단어 ${addedTrainingWords}개 추가` : ""}` +
    (omittedTrainingWords ? ` · 어휘 초과 단어 ${omittedTrainingWords}개 제외` : "");
  document.querySelector("#training-loss").textContent = "";
  const losses = [];
  let step = 0;

  const runStep = async () => {
    if (cancelTransformerTraining) {
      setTransformerTrainingControls(false);
      document.querySelector("#training-status").textContent = `훈련 중지 · ${step}/${steps}단계 완료`;
      setTransformerModel(transformerModel, "부분 훈련됨");
      await saveTrainingCheckpoint(true);
      return;
    }
    try {
      losses.push(window.OCTO_MINI_GPT.trainStep(transformerModel, corpus));
      step += 1;
      if (step % TRAINING_CHECKPOINT_INTERVAL === 0 && step < steps) {
        await saveTrainingCheckpoint(false);
      }
      const start = Math.max(0, losses.length - 25);
      const recentLoss = losses.slice(start).reduce((sum, loss) => sum + loss, 0) / (losses.length - start);
      document.querySelector("#training-progress-fill").style.width = `${(step / steps) * 100}%`;
      document.querySelector("#training-status").textContent =
        `훈련 중 · ${step}/${steps}단계 · 어휘 적용률 ${vocabularyCoverage}%` +
        (addedTrainingWords ? ` · 새 단어 ${addedTrainingWords}개` : "") +
        (omittedTrainingWords ? ` · 한도 초과 ${omittedTrainingWords}개` : "");
      document.querySelector("#training-loss").textContent = `최근 손실 ${recentLoss.toFixed(3)}`;
      if (step < steps) {
        window.setTimeout(runStep, 0);
      } else {
        setTransformerTrainingControls(false);
        setTransformerModel(transformerModel, "훈련 완료");
        document.querySelector("#training-status").textContent = `훈련 완료 · ${steps}단계 · 학습률 0.01`;
        const saved = await saveTrainingCheckpoint(true);
        showToast(saved
          ? "훈련이 완료됐고 기존 가중치에 이어 학습한 모델을 저장했습니다."
          : "훈련은 완료됐지만 자동 저장에 실패했습니다. 오류를 확인하고 가중치를 내려받으세요.");
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
    const generated = window.OCTO_MINI_GPT.generate(transformerModel, prompt, 40, 0.8);
    const ignoredPromptCharacters = window.OCTO_MINI_GPT.tokenize(transformerModel, prompt).unknownCharacters;
    document.querySelector("#generation-output").textContent = ignoredPromptCharacters
      ? `${generated}\n\n(모델 어휘에 없는 시작 문자는 ${ignoredPromptCharacters}개 제외했습니다.)`
      : generated;
  } catch (error) {
    console.error("Could not generate a mini-transformer sample.", error);
    showToast(error.message || "문장 생성에 실패했습니다.");
  }
});

async function makeWeightsPackage() {
  if (!transformerModel) throw new Error("먼저 모델을 초기화하거나 훈련하세요.");
  window.OCTO_MINI_GPT.validateModel(transformerModel);
  if (!crypto.subtle || typeof crypto.subtle.digest !== "function") {
    throw new Error("SHA-256 무결성 검사를 지원하는 보안 연결(HTTPS)에서 다시 시도하세요.");
  }
  
  // 가중치를 int8로 양자화
  const { quantized, scales } = window.OCTO_MINI_GPT.quantizeWeights(transformerModel.weights);
  const quantizedModel = {
    format: transformerModel.format,
    language: transformerModel.language,
    tokenizer: transformerModel.tokenizer,
    endings: transformerModel.endings,
    vocabulary: transformerModel.vocabulary,
    config: transformerModel.config,
    weights: quantized,
    scales: scales,  // 역양자화 스케일 저장
  };
  
  const modelJSON = JSON.stringify(quantizedModel);
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(modelJSON));
  const checksum = [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, "0")).join("");
  return {
    format: WEIGHTS_PACKAGE_FORMAT,
    checksum: { algorithm: "SHA-256", value: checksum },
    model: quantizedModel,
  };
}

async function readWeightsPackage(json) {
  if (typeof json !== "string" || json.length > MAX_WEIGHTS_FILE_BYTES) {
    throw new Error("가중치 JSON이 256 MiB 제한을 넘었습니다.");
  }
  const byteLength = new TextEncoder().encode(json).byteLength;
  if (byteLength > MAX_WEIGHTS_FILE_BYTES) {
    throw new Error("가중치 파일이 256 MiB 제한을 넘었습니다.");
  }
  const parsed = JSON.parse(json);
  const packageKeys = Object.keys(parsed || {}).sort();
  if (packageKeys.join(",") !== "checksum,format,model") {
    throw new Error("가중치 패키지에 알 수 없거나 누락된 항목이 있습니다.");
  }
  
  // ===== 버전 호환성 처리 =====
  // 기존 float32 v3 가중치를 v4로 자동 변환
  let targetFormat = WEIGHTS_PACKAGE_FORMAT;
  if (parsed.format === "octo-mini-weights-package-v3") {
    console.warn("기존 float32 v3 가중치를 감지했습니다. 자동으로 int8 v4로 변환 중...");
    targetFormat = "octo-mini-weights-package-v3";  // 호환성 검증 스킵
  } else if (parsed.format !== WEIGHTS_PACKAGE_FORMAT) {
    throw new Error(`지원하지 않는 가중치 파일입니다 (v${parsed.format}). int8 v4 패키지를 사용하세요.`);
  }
  
  if (!parsed || 
      !parsed.checksum || parsed.checksum.algorithm !== "SHA-256" ||
      typeof parsed.checksum.value !== "string" || !/^[a-f0-9]{64}$/u.test(parsed.checksum.value)) {
    throw new Error("지원하지 않는 가중치 파일입니다. Octo 가중치 패키지(int8 v4)를 사용하세요.");
  }
  const checksumKeys = Object.keys(parsed.checksum).sort();
  if (checksumKeys.join(",") !== "algorithm,value") {
    throw new Error("체크섬 정보에 알 수 없거나 누락된 항목이 있습니다.");
  }
  if (!crypto.subtle || typeof crypto.subtle.digest !== "function") {
    throw new Error("SHA-256 무결성 검사를 지원하는 보안 연결(HTTPS)에서 다시 시도하세요.");
  }
  
  // int8로 양자화된 가중치를 float32로 역양자화
  let model = parsed.model;
  if (model.scales) {
    // 양자화된 모델이므로 역양자화
    const dequantized = window.OCTO_MINI_GPT.dequantizeWeights(model.weights, model.scales);
    model = {
      format: model.format,
      language: model.language,
      tokenizer: model.tokenizer,
      endings: model.endings,
      vocabulary: model.vocabulary,
      config: model.config,
      weights: dequantized,
      // scales는 저장하지 않음 (메모리 절약)
    };
  }
  
  model = window.OCTO_MINI_GPT.validateModel(model);
  const modelJSON = JSON.stringify(parsed.model);  // 원본 양자화 모델로 검증
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(modelJSON));
  const actualChecksum = [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, "0")).join("");
  if (actualChecksum !== parsed.checksum.value) {
    throw new Error("가중치 SHA-256 검증에 실패했습니다. 파일이 손상되었거나 변경되었을 수 있습니다.");
  }
  return model;
}

document.querySelector("#export-transformer-weights").addEventListener("click", async () => {
  try {
    const packageData = await makeWeightsPackage();
    const json = JSON.stringify(packageData, null, 2);
    if (new TextEncoder().encode(json).byteLength > MAX_WEIGHTS_FILE_BYTES) {
      throw new Error("가중치 패키지가 256 MiB 제한을 넘었습니다.");
    }
    document.querySelector("#transformer-weights").value = json;
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "octo-mini-decoder-weights.json";
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    showToast("SHA-256 검증값을 포함한 가중치 JSON을 다운로드했습니다.");
  } catch (error) {
    console.error("Could not export mini-transformer weights.", error);
    showToast(error.message || "가중치 내보내기에 실패했습니다.");
  }
});

document.querySelector("#copy-transformer-weights").addEventListener("click", async () => {
  try {
    const packageData = await makeWeightsPackage();
    const json = JSON.stringify(packageData, null, 2);
    if (new TextEncoder().encode(json).byteLength > MAX_WEIGHTS_FILE_BYTES) {
      throw new Error("가중치 패키지가 256 MiB 제한을 넘었습니다.");
    }
    document.querySelector("#transformer-weights").value = json;
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

document.querySelector("#import-transformer-weights").addEventListener("click", async () => {
  try {
    await importTransformerWeights(document.querySelector("#transformer-weights").value);
  } catch (error) {
    console.error("Could not import mini-transformer weights.", error);
    showToast(error.message || "가중치 JSON을 불러오지 못했습니다.");
  }
});

async function importTransformerWeights(json) {
  const imported = await readWeightsPackage(json);
  const addedTokens = setTransformerModel(imported, "JSON 불러옴");
  document.querySelector("#training-status").textContent = addedTokens
    ? `${addedTokens}개 기본 토큰을 보충했습니다. 기존 학습값은 보존했으며 새 토큰은 가능한 기존 문자 가중치로 초기화하고, 없는 문자는 무작위 초기화했으므로 관련 문장으로 학습한 뒤 내보내세요.`
    : "가중치를 불러왔습니다. 같은 어휘에 포함된 문장으로 이어서 훈련하거나 출력을 생성할 수 있습니다.";
  await saveTrainingCheckpoint(false);
  document.querySelector("#generation-output").textContent = "가중치를 불러왔습니다. 시작 문구를 입력하고 생성을 눌러 보세요.";
  showToast(addedTokens
  ? `검증된 가중치를 보존하고 기본 토큰 ${addedTokens}개를 보충했습니다. 새 토큰을 훈련하고 가중치를 다시 내려받으세요.`
    : "SHA-256 및 모델 구조 검증을 통과한 가중치를 불러왔습니다.");
}

document.querySelector("#connect-weights-file").addEventListener("click", async () => {
  try {
    if (typeof window.showOpenFilePicker !== "function") {
      throw new Error("이 브라우저는 파일 직접 저장을 지원하지 않습니다. Chrome 또는 Edge에서 HTTPS로 열어 주세요.");
    }
    const [handle] = await window.showOpenFilePicker({
      multiple: false,
      types: [{ description: "Octo 가중치 JSON", accept: { "application/json": [".json"] } }],
    });
    const permission = await handle.requestPermission({ mode: "readwrite" });
    if (permission !== "granted") throw new Error("선택한 파일에 대한 쓰기 권한이 허용되지 않았습니다.");
    const file = await handle.getFile();
    if (file.size > MAX_WEIGHTS_FILE_BYTES) throw new Error("가중치 파일이 256 MiB 제한을 넘었습니다.");
    const json = await file.text();
    const imported = await readWeightsPackage(json);
    weightsFileHandle = handle;
    const addedTokens = setTransformerModel(imported, "weights.json 연결됨");
    document.querySelector("#training-status").textContent = addedTokens
      ? `${addedTokens}개 기본 토큰을 보충했습니다. 기존 학습값은 유지되고 새 토큰은 기존 문자 가중치(없는 문자는 무작위 값)로 초기화되므로 관련 문장으로 훈련해야 합니다.`
      : "선택한 weights.json의 가중치를 불러왔습니다. 다음 훈련은 이 가중치에서 이어집니다.";
    const checkpointSaved = await saveTrainingCheckpoint(false);
    if (checkpointSaved) {
      setWeightsSaveStatus(addedTokens
        ? "확장된 가중치를 브라우저 체크포인트에 저장했습니다. 선택한 weights.json 파일은 훈련 완료 전까지 바뀌지 않습니다."
        : "브라우저 체크포인트를 갱신했고, 선택한 로컬 weights.json에도 훈련 완료 시 결과를 기록합니다.");
    }
    showToast("가중치 파일을 검증하고 연결했습니다.");
  } catch (error) {
    if (error.name === "AbortError") return;
    console.error("Could not connect weights file.", error);
    showToast(error.message || "가중치 파일을 연결하지 못했습니다.");
  }
});

document.querySelector("#select-transformer-weights").addEventListener("click", () => {
  document.querySelector("#transformer-weights-file").click();
});

document.querySelector("#transformer-weights-file").addEventListener("change", async event => {
  const [file] = event.target.files || [];
  if (!file) return;
  try {
    if (file.size > MAX_WEIGHTS_FILE_BYTES) {
      throw new Error("가중치 파일이 256 MiB 제한을 넘었습니다.");
    }
    const json = await file.text();
    document.querySelector("#transformer-weights").value = json;
    await importTransformerWeights(json);
  } catch (error) {
    console.error("Could not load mini-transformer weights file.", error);
    showToast(error.message || "가중치 파일을 불러오지 못했습니다.");
  } finally {
    event.target.value = "";
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
      const isValidEntry = word => !((category === "verbs" || category === "adjectives") && !word.endsWith("-"));
      const wordList = [definition.items, definition.additional || "", definition.more || ""].filter(Boolean).join(" ");
      const baseEntries = wordList.trim().split(/\s+/u)
        .filter(word => word && isValidEntry(word))
        .map(word => ({ word, group: "" }));
      const existingWords = new Set(baseEntries.map(entry => entry.word));
      const dailyEntries = [];
      for (const word of (definition.daily || "").trim().split(/\s+/u)) {
        if (word && isValidEntry(word) && !existingWords.has(word)) {
          existingWords.add(word);
          dailyEntries.push({ word, group: "" });
          if (dailyEntries.length === 300) break;
        }
      }
      entries.push(...dailyEntries, ...baseEntries);
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
      entries: uniqueEntries.slice(0, 800),
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
makeMessage("assistant", "안녕하세요! 자동 모드에서는 저장된 자료에 직접적인 근거가 있을 때 출처와 함께 답하고, 그 외의 일상 대화는 로컬 미니 모델로 생성해요. 모델이 작아 답변이 부정확할 수 있습니다.", [], "system");
transformerReady = initializeSavedTransformer();


// ===== 자동 훈련 시스템 =====

// ===== 사용자 정의 한국어 문장 데이터베이스 =====
const userTrainingCorpus = [
  // 여기에 한국어 문장을 추가하세요
  // 예시:
  // "안녕하세요. 반갑습니다.",
  // "오늘 날씨가 정말 좋네요.",
];

// 모바일 감지
const isMobile = () => /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

const autoTrainer = {
  isRunning: false,
  isPaused: false,
  totalSteps: 0,
  currentStep: 0,
  losses: [],
  trainingTimeoutId: null,
  lastSaveStep: 0,
  sessionStartTime: 0,
  
  // 모바일/데스크톱별 최적화 설정
  getOptimizations() {
    const mobile = isMobile();
    return {
      isMobile: mobile,
      delayBetweenSteps: mobile ? 50 : 5,      // 모바일은 더 긴 지연
      saveInterval: mobile ? 50 : 25,          // 모바일은 더 자주 저장
      maxLossesBuffer: mobile ? 10 : 50,       // 모바일은 메모리 절약
      checkpointInterval: mobile ? 300000 : 600000,  // 모바일: 5분, 데스크톱: 10분
    };
  },
  
  async start() {
    if (!transformerModel) {
      showToast("먼저 모델을 초기화하거나 훈련하세요.");
      return;
    }
    
    // 사용자 문장 데이터 확인
    if (userTrainingCorpus.length === 0) {
      showToast("먼저 한국어 문장을 추가하세요. (자동 훈련 설정 참고)");
      return;
    }
    
    this.isRunning = true;
    this.isPaused = false;
    this.totalSteps = parseInt(document.querySelector("#auto-training-steps").value) || 100;
    this.currentStep = 0;
    this.losses = [];
    this.lastSaveStep = 0;
    this.sessionStartTime = Date.now();
    
    updateAutoTrainerUI();
    showToast("자동 훈련 시작...");
    
    // 비동기 훈련 루프 (무한 루프 대신 개별 단계 실행)
    this.runNextStep();
  },
  
  async runNextStep() {
    const opts = this.getOptimizations();
    
    // 중지되었거나 완료됨
    if (!this.isRunning || this.currentStep >= this.totalSteps) {
      if (this.currentStep >= this.totalSteps) {
        showToast("자동 훈련 완료!");
        await saveTrainingCheckpoint(true);
      }
      this.isRunning = false;
      updateAutoTrainerUI();
      return;
    }
    
    // 일시정지 중
    if (this.isPaused) {
      updateAutoTrainerUI();
      this.trainingTimeoutId = setTimeout(() => this.runNextStep(), 500);
      return;
    }
    
    try {
      // 사용자 데이터에서 랜덤 문장 선택
      const sentence = userTrainingCorpus[Math.floor(Math.random() * userTrainingCorpus.length)];
      
      // 훈련 실행
      const loss = window.OCTO_MINI_GPT.trainStep(transformerModel, sentence, 512);
      this.losses.push(loss);
      
      // 손실 버퍼 관리 (메모리 절약)
      if (this.losses.length > opts.maxLossesBuffer) {
        this.losses.shift();
      }
      
      this.currentStep += 1;
      
      // 저장 실행
      if (this.currentStep - this.lastSaveStep >= opts.saveInterval) {
        await saveTrainingCheckpoint(false);
        this.lastSaveStep = this.currentStep;
      }
      
      // 정기적 체크포인트 (배터리 절약, 모바일 백그라운드 대비)
      const elapsedTime = Date.now() - this.sessionStartTime;
      if (elapsedTime > opts.checkpointInterval) {
        await saveTrainingCheckpoint(true);
        this.sessionStartTime = Date.now();
      }
      
      // UI 업데이트
      updateAutoTrainerUI();
      
    } catch (error) {
      console.error("훈련 오류:", error);
      showToast(`훈련 오류: ${error.message}`);
      this.stop();
      return;
    }
    
    // 다음 단계 예약 (requestIdleCallback 우선, 폴백 setTimeout)
    const opts2 = this.getOptimizations();
    if (typeof requestIdleCallback === "function" && !opts2.isMobile) {
      this.trainingTimeoutId = requestIdleCallback(() => this.runNextStep(), { timeout: 1000 });
    } else {
      this.trainingTimeoutId = setTimeout(() => this.runNextStep(), opts2.delayBetweenSteps);
    }
  },
  
  pause() {
    this.isPaused = !this.isPaused;
    updateAutoTrainerUI();
    showToast(this.isPaused ? "훈련 일시정지" : "훈련 재개");
  },
  
  stop() {
    this.isRunning = false;
    this.isPaused = false;
    if (this.trainingTimeoutId) {
      clearTimeout(this.trainingTimeoutId);
      cancelIdleCallback(this.trainingTimeoutId);
      this.trainingTimeoutId = null;
    }
    updateAutoTrainerUI();
    showToast("자동 훈련 중지됨");
  },
  
  async downloadJSON() {
    if (!transformerModel) {
      showToast("모델이 없습니다.");
      return;
    }
    
    try {
      const packageData = await makeWeightsPackage();
      const json = JSON.stringify(packageData, null, 2);
      
      const blob = new Blob([json], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `octo-weights-${new Date().toISOString().slice(0, 10)}.json`;
      link.click();
      URL.revokeObjectURL(url);
      
      showToast("가중치 JSON을 다운로드했습니다.");
    } catch (error) {
      showToast(`다운로드 오류: ${error.message}`);
    }
  },
  
  async copyJSON() {
    if (!transformerModel) {
      showToast("모델이 없습니다.");
      return;
    }
    
    try {
      const packageData = await makeWeightsPackage();
      const json = JSON.stringify(packageData, null, 2);
      
      await navigator.clipboard.writeText(json);
      showToast("가중치 JSON을 클립보드에 복사했습니다.");
    } catch (error) {
      showToast(`복사 오류: ${error.message}`);
    }
  }
};

function updateAutoTrainerUI() {
  const startBtn = document.querySelector("#auto-training-start");
  const pauseBtn = document.querySelector("#auto-training-pause");
  const stopBtn = document.querySelector("#auto-training-stop");
  const downloadBtn = document.querySelector("#auto-training-download");
  const copyBtn = document.querySelector("#auto-training-copy");
  const progressDiv = document.querySelector("#auto-training-progress");
  const statusDiv = document.querySelector("#auto-training-status");
  
  if (!startBtn) return; // UI 요소가 없으면 리턴
  
  // 버튼 상태
  startBtn.disabled = autoTrainer.isRunning;
  pauseBtn.disabled = !autoTrainer.isRunning;
  stopBtn.disabled = !autoTrainer.isRunning;
  
  // 진행률 표시
  const progress = autoTrainer.totalSteps > 0 
    ? Math.round((autoTrainer.currentStep / autoTrainer.totalSteps) * 100) 
    : 0;
  
  progressDiv.style.width = progress + "%";
  progressDiv.textContent = progress + "%";
  
  // 상태 텍스트
  if (autoTrainer.isRunning) {
    const recentLosses = autoTrainer.losses.slice(-5);
    const avgLoss = recentLosses.length > 0
      ? (recentLosses.reduce((a, b) => a + b) / recentLosses.length).toFixed(4)
      : "계산 중...";
    
    const lastLoss = recentLosses.length > 0
      ? recentLosses[recentLosses.length - 1].toFixed(4)
      : "계산 중...";
    
    const opts = autoTrainer.getOptimizations();
    const deviceType = opts.isMobile ? "📱 모바일" : "💻 데스크톱";
    
    statusDiv.innerHTML = `
      <div style="font-size: 13px; color: #666;">
        <div>${deviceType} · 단계: ${autoTrainer.currentStep}/${autoTrainer.totalSteps}</div>
        <div>최근 손실: ${lastLoss}</div>
        <div>${autoTrainer.isPaused ? "⏸️ 일시정지 중" : "▶️ 훈련 중"}</div>
      </div>
    `;
  } else {
    statusDiv.innerHTML = `
      <div style="font-size: 13px; color: #666;">
        <div>최종 단계: ${autoTrainer.currentStep}/${autoTrainer.totalSteps}</div>
        <div>${autoTrainer.losses.length > 0 
          ? `최종 손실: ${(autoTrainer.losses[autoTrainer.losses.length - 1]).toFixed(4)}`
          : "준비됨"}</div>
      </div>
    `;
  }
}

// 자동 훈련 UI 초기화
function initAutoTrainerUI() {
  const trainingPanel = document.querySelector(".transformer-panel");
  if (!trainingPanel) return;
  
  // 자동 훈련 섹션 HTML
  const autoTrainerHTML = `
    <div style="margin-top: 30px; padding: 15px; border-top: 2px solid #eee;">
      <h3 style="margin: 0 0 15px 0; font-size: 16px; color: #333;">🤖 자동 훈련</h3>
      
      <div style="display: grid; gap: 10px;">
        <div>
          <label style="font-size: 13px; color: #666; font-weight: bold;">📝 한국어 문장 추가</label>
          <textarea 
            id="auto-training-corpus"
            placeholder="한국어 문장을 입력하세요. 한 문장씩 줄바꿈으로 구분합니다.&#10;예시:&#10;안녕하세요. 반갑습니다.&#10;오늘 날씨가 정말 좋네요.&#10;한국어를 공부합니다."
            style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; font-family: monospace; font-size: 12px; height: 100px; resize: vertical;"
          ></textarea>
          <button 
            onclick="addCorpusFromTextarea()"
            type="button"
            style="width: 100%; padding: 8px; margin-top: 8px; background: #6c757d; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;"
          >✅ 문장 추가</button>
          <p style="font-size: 12px; color: #999; margin: 8px 0 0 0;" id="corpus-count">
            추가된 문장: 0개
          </p>
        </div>
        
        <div>
          <label for="auto-training-steps" style="font-size: 13px; color: #666; font-weight: bold;">⚙️ 훈련 단계:</label>
          <input 
            type="number" 
            id="auto-training-steps" 
            value="100" 
            min="1" 
            max="10000"
            style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box;"
          >
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
          <button 
            id="auto-training-start" 
            type="button"
            onclick="autoTrainer.start()"
            style="padding: 10px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;"
          >▶️ 시작</button>
          <button 
            id="auto-training-pause" 
            type="button"
            onclick="autoTrainer.pause()"
            disabled
            style="padding: 10px; background: #ffc107; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; opacity: 0.5;"
          >⏸️ 일시정지</button>
        </div>
        
        <button 
          id="auto-training-stop" 
          type="button"
          onclick="autoTrainer.stop()"
          disabled
          style="padding: 10px; background: #dc3545; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; opacity: 0.5;"
        >⏹️ 중지</button>
        
        <div id="auto-training-progress-container" style="background: #f0f0f0; border-radius: 4px; overflow: hidden; height: 24px; position: relative;">
          <div 
            id="auto-training-progress"
            style="height: 100%; background: linear-gradient(90deg, #007bff, #0056b3); width: 0%; display: flex; align-items: center; justify-content: center; color: white; font-size: 12px; font-weight: bold; transition: width 0.3s ease;"
          >0%</div>
        </div>
        
        <div id="auto-training-status" style="font-size: 13px; color: #666; line-height: 1.5;">
          준비됨
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
          <button 
            id="auto-training-download"
            type="button"
            onclick="autoTrainer.downloadJSON()"
            style="padding: 10px; background: #17a2b8; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;"
          >📥 JSON 다운로드</button>
          <button 
            id="auto-training-copy"
            type="button"
            onclick="autoTrainer.copyJSON()"
            style="padding: 10px; background: #6c757d; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold;"
          >📋 JSON 복사</button>
        </div>
      </div>
    </div>
  `;
  
  // HTML 추가
  trainingPanel.insertAdjacentHTML('beforeend', autoTrainerHTML);
}

function addCorpusFromTextarea() {
  const textarea = document.querySelector("#auto-training-corpus");
  if (!textarea) return;
  
  const text = textarea.value.trim();
  if (!text) {
    showToast("문장을 입력하세요.");
    return;
  }
  
  // 줄바꿈으로 문장 분리
  const sentences = text.split('\n')
    .map(s => s.trim())
    .filter(s => s.length > 0);
  
  // 기존 데이터에 추가
  userTrainingCorpus.push(...sentences);
  
  // UI 업데이트
  const countSpan = document.querySelector("#corpus-count");
  if (countSpan) {
    countSpan.textContent = `추가된 문장: ${userTrainingCorpus.length}개`;
  }
  
  // 텍스트 초기화
  textarea.value = '';
  
  showToast(`${sentences.length}개 문장이 추가되었습니다. (총 ${userTrainingCorpus.length}개)`);
}

// 페이지 로드 시 자동 훈련 UI 초기화
window.addEventListener('load', () => {
  setTimeout(initAutoTrainerUI, 500);
});

// ===== 모바일 백그라운드 대응 =====
let pageVisibility = document.visibilityState;

document.addEventListener('visibilitychange', async () => {
  const wasVisible = pageVisibility === 'visible';
  pageVisibility = document.visibilityState;
  const isNowVisible = pageVisibility === 'visible';
  
  if (wasVisible && !isNowVisible) {
    // 백그라운드로 전환: 훈련 일시정지 + 체크포인트 저장
    if (autoTrainer.isRunning && !autoTrainer.isPaused) {
      console.log('[Auto-Trainer] 페이지가 백그라운드로 전환됨. 훈련 일시정지 및 체크포인트 저장.');
      autoTrainer.pause();
      await saveTrainingCheckpoint(true);
    }
  } else if (!wasVisible && isNowVisible) {
    // 포그라운드로 복귀: 훈련 재개 가능 (사용자 확인 후)
    if (autoTrainer.isPaused) {
      console.log('[Auto-Trainer] 페이지가 포그라운드로 복귀. 훈련 일시정지 상태 유지.');
      showToast("훈련이 일시정지되어 있습니다. 재개 버튼을 눌러 계속하세요.");
    }
  }
});

// 페이지 언로드 시 훈련 중지 + 최종 저장
window.addEventListener('beforeunload', async (event) => {
  if (autoTrainer.isRunning) {
    // 훈련 중지
    autoTrainer.stop();
    // 최종 체크포인트 저장 시도 (비동기이지만 기다리지 않음)
    saveTrainingCheckpoint(true).catch(err => console.error('Final checkpoint save failed:', err));
  }
});

// ===== 메모리 경고 대응 (Apple Safari, Chrome Android) =====
if ('memory' in performance) {
  // 메모리 모니터링 (Chrome 개발자 도구)
  setInterval(() => {
    const memUsage = performance.memory;
    if (memUsage && memUsage.usedJSHeapSize / memUsage.jsHeapSizeLimit > 0.85) {
      console.warn('[Auto-Trainer] 메모리 사용률이 높습니다:', 
        Math.round((memUsage.usedJSHeapSize / memUsage.jsHeapSizeLimit) * 100) + '%');
      
      // 훈련 중이고 메모리가 부족하면 일시정지 + 저장
      if (autoTrainer.isRunning && !autoTrainer.isPaused) {
        console.warn('[Auto-Trainer] 메모리 부족으로 일시정지합니다.');
        autoTrainer.pause();
        saveTrainingCheckpoint(true);
        showToast("메모리 부족으로 훈련이 일시정지되었습니다. 재개하거나 저장하세요.");
      }
    }
  }, 5000);
}

// ===== 온라인/오프라인 대응 =====
window.addEventListener('offline', () => {
  console.log('[Auto-Trainer] 오프라인 전환');
  if (autoTrainer.isRunning) {
    autoTrainer.pause();
    showToast("⚠️ 오프라인 상태. 훈련이 일시정지되었습니다.");
  }
});

window.addEventListener('online', () => {
  console.log('[Auto-Trainer] 온라인 복귀');
  showToast("✅ 온라인 복귀. 훈련을 재개할 수 있습니다.");
});
