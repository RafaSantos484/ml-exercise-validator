# 🏋️ ML Exercise Validator

Solução web que utiliza visão computacional e aprendizado de máquina para validar automaticamente a execução de exercícios físicos, fornecendo feedback em tempo real diretamente no navegador do usuário.

<img src="https://github.com/user-attachments/assets/a83bb939-02b8-406b-a3dd-dc5cbd09f694" height="400" alt="incorrect execution">
<img src="https://github.com/user-attachments/assets/60e3656e-42d0-4e07-bebe-921e3cb7efcd" height="400" alt="correct execution">

## 🌐 Teste Online

Experimente a aplicação em tempo real: [ML Exercise Validator](https://ml-exercise-validator.vercel.app/)

## 📸 Visão Geral

Este projeto permite que usuários realizem exercícios físicos, como a prancha alta, enquanto recebem validações automáticas sobre a correção da postura. A validação é feita com base na detecção de pontos do corpo utilizando o BlazePose (MediaPipe) e modelos de inferência treinados previamente, como KNN, SVM, Random Forest, regressão logística, além de um modelo empírico baseado em regras como intervalos em que os ângulos se encontram. Os modelos de ML treinados em Python podem ser carregados no ambiente JavaScript dos navegadores utilizando o formato [ONNX](https://onnx.ai/).

## 🚀 Tecnologias Utilizadas

- React
- TypeScript
- MediaPipe
- onnxruntime-web
- WebAssembly (WASM)
- HTML5 Video API

## 🧩 Software design

![Class diagram](https://github.com/user-attachments/assets/200a6963-d85f-4e9a-b078-771bac0ca454)

Os principais design patterns usados foram:
- Factory Method
- Singleton
- Strategy
- Facade
