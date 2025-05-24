    // Função para escapar caracteres HTML
      function escapeHtml(unsafe) {
        return unsafe
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;")
          .replace(/'/g, "&#039;");
      }

      document.addEventListener("DOMContentLoaded", function () {
        const categorySelect = document.getElementById("category");
        const options = categorySelect.options;
        const randomIndex = Math.floor(Math.random() * options.length);

        // Define uma opção aleatória
        categorySelect.selectedIndex = randomIndex;

        // Inicializa as vozes
        initVoices();
      });

      // Funções auxiliares para desabilitar/habilitar botões e mostrar/ocultar mensagens
      function disableButtons() {
        document.getElementById("speakButton").disabled = true;
        document.getElementById("generateButton").disabled = true;
      }

      function enableButtons() {
        document.getElementById("speakButton").disabled = false;
        document.getElementById("generateButton").disabled = false;
      }

      function showMessage(message) {
        let statusMessage = document.getElementById("statusMessage");
        statusMessage.innerText = message;
        statusMessage.style.display = "block";
      }

      function hideMessage() {
        let statusMessage = document.getElementById("statusMessage");
        statusMessage.style.display = "none";
      }

      document.getElementById("text").addEventListener("input", function () {
        fetchPronunciation(this.value);
      });

      function fetchPronunciation(text) {
        fetch("/pronounce", {
          method: "POST",
          headers: {
            "Content-Type": "application/x-www-form-urlencoded",
          },
          body: "text=" + encodeURIComponent(text),
        })
          .then((response) => {
            if (!response.ok) {
              // Se o servidor retornar, por exemplo, 500 ou outro erro
              throw new Error(
                "Erro no servidor /pronounce: " + response.status
              );
            }
            return response.json();
          })
          .then((data) => {
            // Verifica se a resposta contém um campo "error"
            if (data.error) {
              console.error(
                "Erro retornado pelo servidor /pronounce:",
                data.error
              );
              // Aqui você pode exibir uma mensagem de erro na tela, se quiser
              return; // Encerra a função, não prossegue
            }

            // Verifica se data.pronunciations existe
            if (!data.pronunciations) {
              console.warn("pronunciations não veio na resposta. data =", data);
              return; // Encerra a função
            }

            // Agora sim podemos usar .split
            const pronunciations = data.pronunciations;
            document.getElementById("pronunciation").innerText = pronunciations;
          })
          .catch((error) => {
            console.error("Erro ao buscar pronúncia:", error);
          });

        // Busca dicas de pronúncia
        fetch("/hints", {
          method: "POST",
          headers: {
            "Content-Type": "application/x-www-form-urlencoded",
          },
          body: "text=" + encodeURIComponent(text),
        })
          .then((response) => response.json())
          .then((data) => {
            if (data.error) {
              console.error("Erro retornado pelo servidor /hints:", data.error);
              return;
            }

            const hintsList = document.getElementById("hintsList");
            hintsList.innerHTML = ""; // Limpa a lista

            if (data.hints && data.hints.length > 0) {
              data.hints.forEach((hint) => {
                const li = document.createElement("li");
                // Importante: Usar innerHTML em vez de innerText para renderizar as tags HTML
                li.innerHTML = `<strong>${escapeHtml(
                  hint.word
                )}</strong>: ${hint.explanations.join(", ")}`;
                hintsList.appendChild(li);
              });
            } else {
              hintsList.innerHTML = "<li>Nenhuma dica disponível.</li>";
            }
          })
          .catch((error) => {
            console.error("Erro ao buscar dicas:", error);
          });
      }

      function generateSentence() {
        disableButtons();
        showMessage("Gerando frase...");

        const category = document.getElementById("category").value;

        fetch("/get_sentence", {
          method: "POST",
          headers: {
            "Content-Type": "application/x-www-form-urlencoded",
          },
          body: "category=" + encodeURIComponent(category),
        })
          .then((response) => response.json())
          .then((data) => {
            if (data.error) {
              console.error(
                "Erro retornado pelo servidor /get_sentence:",
                data.error
              );
              showMessage("Erro ao gerar frase. Tente novamente.");
              return;
            }

            document.getElementById("text").value = data.fr_sentence;
            fetchPronunciation(data.fr_sentence);
            hideMessage();
          })
          .catch((error) => {
            console.error("Erro ao gerar frase:", error);
            showMessage("Erro ao gerar frase. Tente novamente.");
          })
          .finally(() => {
            enableButtons();
          });
      }

      let frenchVoices = [];
      let selectedVoice = null;
      let synth = window.speechSynthesis;
      let utterance = null;

      // Inicializa as vozes e tenta selecionar a melhor voz francesa
      function initVoices() {
        // Tenta obter as vozes imediatamente
        frenchVoices = synth
          .getVoices()
          .filter((voice) => voice.lang.startsWith("fr"));
        console.log("Vozes iniciais:", frenchVoices.length);

        // Se não houver vozes, espera pelo evento 'voiceschanged'
        if (frenchVoices.length === 0 && "onvoiceschanged" in synth) {
          synth.onvoiceschanged = function () {
            console.log("Evento voiceschanged disparado");
            frenchVoices = synth
              .getVoices()
              .filter((voice) => voice.lang.startsWith("fr"));
            console.log("Vozes carregadas após evento:", frenchVoices.length);
            selectedVoice = getBestFrenchVoice();
            if (selectedVoice) {
              console.log(
                "Melhor voz francesa selecionada após evento:",
                selectedVoice.name,
                selectedVoice.lang
              );
            } else {
              console.warn(
                "Nenhuma voz francesa de alta qualidade encontrada após evento."
              );
            }
          };
        } else {
          selectedVoice = getBestFrenchVoice();
          if (selectedVoice) {
            console.log(
              "Melhor voz francesa selecionada inicialmente:",
              selectedVoice.name,
              selectedVoice.lang
            );
          } else {
            console.warn(
              "Nenhuma voz francesa de alta qualidade encontrada inicialmente."
            );
          }
        }

        // Fallback para garantir que as vozes sejam carregadas em alguns navegadores
        setTimeout(() => {
          if (frenchVoices.length === 0) {
            console.log("Tentando obter vozes após timeout...");
            frenchVoices = synth
              .getVoices()
              .filter((voice) => voice.lang.startsWith("fr"));
            console.log("Vozes carregadas após timeout:", frenchVoices.length);
            selectedVoice = getBestFrenchVoice();
            if (selectedVoice) {
              console.log(
                "Melhor voz francesa selecionada após timeout:",
                selectedVoice.name,
                selectedVoice.lang
              );
            } else {
              console.warn(
                "Nenhuma voz francesa de alta qualidade encontrada após timeout."
              );
            }
          }
        }, 500); // Atraso de 500ms
      }

      // Função para selecionar a melhor voz francesa disponível
      function getBestFrenchVoice() {
        if (frenchVoices.length === 0) {
          console.warn("Nenhuma voz francesa disponível ainda.");
          // Tenta obter novamente, pode ser que o evento não tenha disparado
          frenchVoices = synth
            .getVoices()
            .filter((voice) => voice.lang.startsWith("fr"));
          if (frenchVoices.length === 0) return null;
        }

        // Prioridade para vozes premium/naturais
        const premiumKeywords = [
          "enhanced",
          "premium",
          "neural",
          "google",
          "microsoft",
          "apple",
          "natural",
        ];
        let bestVoice = null;

        // 1. Tenta encontrar vozes premium/naturais específicas para fr-FR
        bestVoice = frenchVoices.find(
          (voice) =>
            voice.lang === "fr-FR" &&
            premiumKeywords.some((keyword) =>
              voice.name.toLowerCase().includes(keyword)
            )
        );
        if (bestVoice) {
          console.log("Voz premium fr-FR encontrada:", bestVoice.name);
          return bestVoice;
        }

        // 2. Tenta encontrar vozes premium/naturais para qualquer variante de francês (fr-CA, etc.)
        bestVoice = frenchVoices.find((voice) =>
          premiumKeywords.some((keyword) =>
            voice.name.toLowerCase().includes(keyword)
          )
        );
        if (bestVoice) {
          console.log(
            "Voz premium francesa (outra variante) encontrada:",
            bestVoice.name
          );
          return bestVoice;
        }

        // 3. Tenta encontrar vozes padrão específicas para fr-FR
        bestVoice = frenchVoices.find(
          (voice) => voice.lang === "fr-FR" && voice.default
        );
        if (bestVoice) {
          console.log("Voz padrão fr-FR encontrada:", bestVoice.name);
          return bestVoice;
        }
        bestVoice = frenchVoices.find((voice) => voice.lang === "fr-FR");
        if (bestVoice) {
          console.log("Voz fr-FR (não padrão) encontrada:", bestVoice.name);
          return bestVoice;
        }

        // 4. Se não encontrou fr-FR, pega a primeira voz francesa padrão disponível
        bestVoice = frenchVoices.find((voice) => voice.default);
        if (bestVoice) {
          console.log(
            "Voz francesa padrão (outra variante) encontrada:",
            bestVoice.name
          );
          return bestVoice;
        }

        // 5. Se não encontrou nenhuma padrão, pega a primeira voz francesa da lista
        if (frenchVoices.length > 0) {
          console.log(
            "Nenhuma voz premium ou padrão encontrada. Usando a primeira voz francesa disponível:",
            frenchVoices[0].name
          );
          return frenchVoices[0];
        }

        console.warn("Nenhuma voz francesa encontrada.");
        return null;
      }

      // Função para falar o texto usando Web Speech API
      function speakText() {
        const text = document.getElementById("text").value;
        if (!text || text.trim() === "") {
          showMessage("Por favor, gere ou digite um texto para escutar.");
          return;
        }

        if (!synth) {
          showMessage("Seu navegador não suporta síntese de voz.");
          return;
        }

        // Cancela qualquer fala anterior
        if (synth.speaking) {
          console.log("Cancelando fala anterior...");
          synth.cancel();
          // Pequeno atraso para garantir que o cancelamento seja processado antes de iniciar nova fala
          setTimeout(() => startSpeaking(text), 100);
        } else {
          startSpeaking(text);
        }
      }

      function startSpeaking(text) {
        // Tenta obter a melhor voz novamente caso a inicialização tenha falhado
        if (!selectedVoice) {
          console.log("Tentando selecionar a melhor voz antes de falar...");
          selectedVoice = getBestFrenchVoice();
        }

        if (!selectedVoice) {
          showMessage(
            "Nenhuma voz em francês encontrada. Verifique as vozes instaladas no seu sistema."
          );
          console.warn("Vozes disponíveis:", synth.getVoices());
          return;
        }

        utterance = new SpeechSynthesisUtterance(text);
        utterance.voice = selectedVoice;
        utterance.lang = selectedVoice.lang; // Garante que o idioma correto seja usado
        utterance.rate = 0.95; // Velocidade ligeiramente mais lenta para clareza
        utterance.pitch = 1.0; // Tom padrão
        utterance.volume = 1.0; // Volume máximo

        utterance.onstart = function () {
          console.log(
            "Iniciando fala com a voz:",
            selectedVoice.name,
            "Idioma:",
            utterance.lang
          );
          showMessage("Falando...");
          disableButtons();
        };

        utterance.onend = function () {
          console.log("Fala concluída.");
          hideMessage();
          enableButtons();
          utterance = null; // Limpa a referência
        };

        utterance.onerror = function (event) {
          console.error("Erro na síntese de voz:", event.error);
          showMessage("Erro ao tentar falar: " + event.error);
          hideMessage();
          enableButtons();
          utterance = null;
        };

        // Workaround para bug do Chrome/Edge que para após 15 segundos
        let intervalId = null;
        if (
          navigator.userAgent.includes("Chrome") ||
          navigator.userAgent.includes("Edg")
        ) {
          intervalId = setInterval(() => {
            if (synth.speaking) {
              console.log("Mantendo a fala ativa (Chrome/Edge workaround)...");
              synth.pause();
              synth.resume();
            } else {
              clearInterval(intervalId);
            }
          }, 14000);

          utterance.addEventListener("end", () => clearInterval(intervalId));
          utterance.addEventListener("error", () => clearInterval(intervalId));
        }

        // Inicia a fala
        console.log("Chamando synth.speak...");
        synth.speak(utterance);
      }
