<p align="center">
  <img src="https://img.shields.io/badge/Apple%20Silicon-Optimized-black?style=for-the-badge&logo=apple" alt="Apple Silicon Optimized">
  <img src="https://img.shields.io/badge/MLX-Native-orange?style=for-the-badge" alt="MLX Native">
  <img src="https://img.shields.io/badge/PyTorch-MPS-red?style=for-the-badge&logo=pytorch" alt="PyTorch MPS">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge" alt="License">
</p>

<h1 align="center">Qwen3-TTS-Mac-GeneLab</h1>

<p align="center">
  Fork de Qwen3-TTS entièrement optimisé pour Apple Silicon Mac<br>
  Double moteur (MLX + PyTorch) pour une expérience TTS native sur Mac
</p>

<p align="center">
  <a href="../README.md">English</a> |
  <a href="README_JA.md">日本語</a> |
  <a href="README_ZH.md">中文</a> |
  <a href="README_KO.md">한국어</a> |
  <a href="README_RU.md">Русский</a> |
  <a href="README_ES.md">Español</a> |
  <a href="README_IT.md">Italiano</a> |
  <a href="README_DE.md">Deutsch</a> |
  **Français** |
  <a href="README_PT.md">Português</a>
</p>

<p align="center">
  <a href="https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab/stargazers">
    <img src="../assets/star.gif" alt="Star this repo!" width="580">
  </a>
  <br>
  <sub>Si ce projet vous est utile, pensez à lui donner une étoile — cela aide beaucoup !</sub>
</p>

---

## Pourquoi Qwen3-TTS-Mac-GeneLab ?

| Fonctionnalité | Qwen3-TTS Officiel | **Ce Projet** |
|----------------|-------------------|---------------|
| Optimisation Apple Silicon | Limitée | **Support complet** |
| Inférence native MLX | Non | **Oui** (quantification 8bit/4bit) |
| PyTorch MPS | Configuration manuelle requise | **Basculement automatique** |
| Interface graphique | Aucune | **Web UI en 10 langues** |
| Clonage vocal | CLI uniquement | **Web UI + transcription automatique Whisper** |
| Gestion de la mémoire | Aucune | **Optimisée pour Unified Memory** |
| Installation | Complexe | **Une seule commande** |

### Innovations clés

1. **Architecture à double moteur**
   - MLX : Natif pour Apple Silicon, quantification 8bit/4bit pour la vitesse et l'efficacité mémoire
   - PyTorch : Basculement automatique pour Voice Clone (exécution CPU float32)

2. **Optimisation automatique basée sur les tâches**
   - CustomVoice -> MLX préféré (rapide)
   - VoiceDesign -> MLX préféré (rapide)
   - VoiceClone -> PyTorch CPU (float32 requis)

3. **Web UI en 10 langues**
   - Interface intuitive basée sur Gradio
   - Changer de langue depuis le menu déroulant en haut

---

## Configuration requise

| Élément | Minimum | Recommandé |
|---------|---------|------------|
| Puce | Apple Silicon (M1) | M2 Pro / M3+ |
| RAM | 16GB | 32GB+ |
| OS | macOS 14 Sonoma | macOS 15 Sequoia |
| Python | 3.10 | 3.11 |
| Espace libre | 10GB | 20GB+ |

> **Vous cherchez la version Windows ?** Consultez [Qwen3-TTS-JP](https://github.com/hiroki-abe-58/Qwen3-TTS-JP) — version native Windows avec support GPU NVIDIA (testé avec RTX 5090).

---

## Démarrage rapide

### 1. Cloner le dépôt

```bash
git clone https://github.com/hiroki-abe-58/Qwen3-TTS-Mac-GeneLab.git
cd Qwen3-TTS-Mac-GeneLab
```

### 2. Configuration (première fois uniquement, ~5-10 min)

```bash
chmod +x setup_mac.sh
./setup_mac.sh
```

### 3. Lancer la Web UI

**Option A : Double-clic (recommandé)**

Double-cliquez sur `run.command` dans le Finder pour lancer automatiquement dans le Terminal.

**Option B : Depuis le terminal**

```bash
./run.sh
```

> Si le port est déjà utilisé, un port disponible est automatiquement détecté.

### 4. Ouvrir dans le navigateur

Ouvrez http://localhost:7860 (vérifiez la sortie du terminal si le port a changé)

---

## Fonctionnalités de la Web UI

### Custom Voice
Générez la parole avec 9 locuteurs prédéfinis. Prend en charge le contrôle des émotions et 10 langues.

### Voice Design
Décrivez les caractéristiques vocales en texte pour générer une voix correspondante.

### Voice Clone
Clonez une voix à partir de seulement 3 secondes d'audio de référence avec transcription automatique Whisper.

> **Note** : Voice Clone nécessite le **modèle Base** (~3.8GB), téléchargé automatiquement lors de la première utilisation.

### Settings
Sélection du moteur (AUTO/MLX/PyTorch), surveillance de la mémoire, gestion des modèles.

---

## Utilisation en CLI

```python
from mac import DualEngine, TaskType
import soundfile as sf

engine = DualEngine()

result = engine.generate(
    text="Hello, this is a voice synthesis demo.",
    task_type=TaskType.CUSTOM_VOICE,
    language="English",
    speaker="Vivian",
)

sf.write("output.wav", result.audio, result.sample_rate)
```

---

## Structure des répertoires

```
Qwen3-TTS-Mac-GeneLab/
├── setup_mac.sh          # Script de configuration
├── run.sh                # Script de lancement (terminal)
├── run.command           # Fichier de lancement (double-clic)
├── pyproject.toml        # Configuration du projet
├── requirements-mac.txt  # Dépendances Mac
├── mac/                  # Code spécifique Mac
│   ├── engine.py         # Gestionnaire double moteur
│   ├── device_utils.py   # Détection de périphérique
│   └── whisper_transcriber.py
├── ui/                   # Gradio Web UI
│   ├── app.py            # Application principale
│   ├── i18n_utils.py     # Utilitaire i18n
│   ├── components/       # Composants des onglets
│   └── i18n/             # 10 fichiers de langue
├── qwen_tts/             # Cœur TTS (upstream)
└── docs/                 # README multilingue
```

---

## Dépannage

| Erreur | Cause | Solution |
|--------|-------|----------|
| `conda not found` | Miniforge non installé | Exécutez `./setup_mac.sh` |
| `No space left on device` | Espace disque insuffisant | Assurez-vous d'avoir 10GB+ libres |
| `RuntimeError: MPS backend` | Opération MPS non supportée | Définissez `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| `Out of memory` | Mémoire insuffisante | Fermez d'autres applications ou utilisez des modèles quantifiés |

---

## Remerciements

- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio)
- [mlx-community](https://huggingface.co/mlx-community)
- [OpenAI Whisper](https://github.com/openai/whisper)

---

## Licence

[Apache License 2.0](../LICENSE)

---

## Contribuer

Les Issues et Pull Requests sont les bienvenues !
