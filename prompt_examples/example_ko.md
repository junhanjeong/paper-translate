## Flamingo: a Visual Language Model for Few-Shot Learning

DeepMind

## Abstract

새로운 task에 대해 소수의 annotated example만으로도 빠르게 적응할 수 있는 모델을 구축하는 것은 멀티모달 머신러닝 연구에서 여전히 해결되지 않은 도전 과제이다. 우리는 이러한 능력을 갖춘 Visual Language Model(VLM) 계열인 **Flamingo**를 소개한다. 우리는 다음과 같은 주요 아키텍처 혁신을 제안한다:
(i) 강력한 사전학습된 vision-only 모델과 language-only 모델을 연결하는 구조,
(ii) 시각 및 텍스트 데이터가 임의로 섞여 있는 시퀀스를 처리할 수 있는 구조,
(iii) 이미지 또는 비디오를 입력으로 자연스럽게 수용할 수 있는 구조.

이러한 유연성 덕분에 Flamingo 모델은 텍스트와 이미지가 임의로 섞여 있는 대규모 멀티모달 웹 코퍼스를 학습에 활용할 수 있으며, 이는 **in-context few-shot learning** 능력을 갖추는 데 핵심적인 요소이다.  
우리는 다양한 이미지 및 비디오 task에 대한 Flamingo 모델의 빠른 적응 능력을 탐구하고 측정하며, 철저한 평가를 수행한다. 여기에는 다음과 같은 open-ended task가 포함된다:

* **Visual Question Answering (VQA)**: 모델이 질문을 받고 그에 대한 답을 생성해야 하는 task,
* **Captioning**: 장면이나 이벤트를 설명하는 능력을 평가하는 task.

또한 다음과 같은 close-ended task도 포함된다:

* **Multiple-choice VQA**: 여러 선택지 중 정답을 고르는 형식의 task.

이 스펙트럼 어디에 위치한 task든지, 단일 Flamingo 모델은 **task-specific 예시를 prompt로 주는 것만으로 few-shot learning 방식으로 새로운 state-of-the-art 성능을 달성**할 수 있다. 많은 벤치마크에서 Flamingo는 수천 배 더 많은 task-specific 데이터로 fine-tuning된 모델보다 더 나은 성능을 보여준다.



![](https://cdn.mathpix.com/cropped/2025_07_26_7c316185968e7585aacbg-02.jpg?height=2242&width=1403&top_left_y=180&top_left_x=361)

Figure 1: Flamingo-80B로부터 얻은 입력과 출력의 선택된 예시. Flamingo는 **few-shot prompting**만으로 다양한 이미지/비디오 이해 task에 빠르게 적응할 수 있다 (상단). 또한 Flamingo는 **별도의 fine-tuning 없이도 multi-image visual dialogue**를 수행할 수 있는 능력을 갖추고 있다 (하단). 더 많은 예시는 Appendix C에 제시되어 있다.

![](https://cdn.mathpix.com/cropped/2025_07_26_7c316185968e7585aacbg-03.jpg?height=475&width=1378&top_left_y=245&top_left_x=371)

Figure 2: Flamingo 결과 개요.
**왼쪽**: 우리의 가장 큰 모델인 Flamingo는 **fine-tuning 없이도** 우리가 다룬 16개의 task 중 6개에서 **state-of-the-art로 fine-tuned된 모델보다 더 뛰어난 성능**을 보인다. 또한, **few-shot 결과가 공개된 9개의 task에서는 Flamingo가 새로운 few-shot state-of-the-art 성능을 기록**한다. 참고로, 16번째 벤치마크인 **RareAct**는 비교할 수 있는 fine-tuned 결과가 없는 zero-shot 벤치마크이므로 생략하였다.
**오른쪽**: Flamingo의 성능은 **모델 크기와 few-shot 예시의 개수가 많아질수록 향상**된다.


## 1 Introduction

지능의 핵심적인 측면 중 하나는 **간단한 지시만으로 새로운 task를 빠르게 학습하는 능력**이다 \[33, 70]. 컴퓨터 비전 분야에서도 이러한 능력에 대한 초기적인 진전이 있었지만, 여전히 가장 널리 사용되는 접근 방식은 **대규모 supervised 데이터로 모델을 사전학습한 뒤, 관심 있는 task에 대해 fine-tuning**을 수행하는 것이다 \[66, 118, 143]. 하지만 이러한 fine-tuning은 **수천 개 이상의 annotated data가 필요**하며, 각 task별로 **세심한 하이퍼파라미터 튜닝**이 요구되고, **많은 자원이 소모**된다는 단점이 있다.  
최근에는 contrastive objective로 학습된 **multimodal vision-language model**들이 등장하면서, **fine-tuning 없이도 새로운 task에 zero-shot으로 적응**하는 것이 가능해졌다 \[50, 85]. 하지만 이러한 모델들은 단순히 **텍스트와 이미지 간의 유사도 점수만 제공**하기 때문에, \*\*미리 정의된 제한된 결과 집합(class label)\*\*이 있는 분류 문제와 같은 제한된 use case에만 사용할 수 있다. 이들은 **언어 생성 능력이 없어**, **captioning이나 visual question answering**과 같은 **open-ended task에는 적합하지 않다**. 이에 반해, 몇몇 연구들은 \*\*시각 정보에 조건을 거는 언어 생성(visual-conditioned language generation)\*\*을 시도했으나 \[17, 114, 119, 124, 132], **소량의 데이터로 학습하는 few-shot setting에서는 성능이 좋지 않았다**.

우리는 이러한 문제를 해결하고자 **Flamingo**를 소개한다. Flamingo는 \*\*Visual Language Model (VLM)\*\*로서, Figure 1에서 보여주듯 **소수의 input/output 예시만으로 prompt를 구성하는 것만으로도, 다양한 open-ended vision-language task에서 새로운 few-shot state-of-the-art 성능**을 달성한다. 우리가 다룬 **16개의 task 중 6개에서는 기존 fine-tuned SOTA보다 더 뛰어난 성능을 보이며**, 이는 Flamingo가 **훨씬 적은 task-specific training data를 사용**함에도 불구하고 달성한 성과이다 (Figure 2 참조).  
이를 가능하게 하기 위해 Flamingo는 \*\*few-shot 학습에서 우수한 성능을 보인 최신 대형 language model (LM)\*\*들의 구조에서 영감을 받았다 \[11, 18, 42, 86]. 이러한 LMs는 텍스트 기반의 interface를 통해 다양한 task를 수행할 수 있으며, **소수의 예시들과 쿼리 입력을 prompt로 주면, 해당 쿼리에 대한 예측 결과를 생성**할 수 있다. 우리는 이러한 방식이 \*\*이미지와 비디오 이해 task들(예: 분류, 캡셔닝, 질문응답 등)\*\*에도 적용될 수 있음을 보인다. 이들 task는 **시각 정보에 기반한 텍스트 생성 문제**로 변환될 수 있다.  
LM과의 차이점은, Flamingo는 **텍스트와 이미지/비디오가 섞여 있는 multimodal prompt를 처리할 수 있어야 한다는 점**이다. Flamingo는 이러한 요구를 충족하는 모델로, **시각 정보를 조건으로 받아들이는 autoregressive text generation model**이다. 즉, **텍스트 token과 이미지/비디오가 섞여 있는 시퀀스를 입력받아 텍스트를 출력할 수 있다.**  
Flamingo는 두 개의 사전학습된 모델을 조합하여 활용한다:

* **시각 정보를 인지(perceive)할 수 있는 vision model**,
* **기초적인 reasoning을 수행할 수 있는 대형 language model**.

이 둘 사이에 **새로운 아키텍처 구성 요소를 삽입하여, 각 모델이 사전학습 동안 축적한 지식을 그대로 유지한 채 연결**되도록 설계되었다.  
또한 Flamingo는 \*\*Perceiver 기반 아키텍처 \[48]\*\*를 통해 **고해상도의 이미지나 비디오도 효율적으로 처리**할 수 있다. 이 구조는 **큰 규모의 시각 입력으로부터 고정된 수의 visual token을 생성**할 수 있어, 다양한 크기의 이미지/비디오 입력을 수용 가능하게 한다.

![](https://cdn.mathpix.com/cropped/2025_07_26_7c316185968e7585aacbg-04.jpg?height=611&width=1380&top_left_y=242&top_left_x=378)

Figure 3: Flamingo 아키텍처 개요.
Flamingo는 **텍스트와 섞여 있는 시각적 데이터를 입력으로 받아 자유형식의 텍스트를 출력**하는 **Visual Language Model (VLM)** 계열의 모델이다.

대형 Language Model (LM)의 성능에 있어 핵심적인 요소 중 하나는 **방대한 양의 텍스트 데이터로 학습되었다는 점**이다. 이러한 학습은 **범용적인 텍스트 생성 능력**을 모델에 부여하며, **task 예시만으로도 뛰어난 성능을 발휘할 수 있게** 해준다. 이와 유사하게, **Flamingo 모델의 학습 방식 또한 최종 성능에 매우 중요한 역할**을 한다는 것을 우리는 실험적으로 보여준다. Flamingo 모델은 **기계학습을 위해 별도로 주석 처리되지 않은, 웹에서 수집한 다양한 대규모 멀티모달 데이터**로 구성된 \*\*신중하게 설계된 데이터 혼합(mixture)\*\*을 이용해 학습된다. 이러한 학습을 거친 후 Flamingo는 **어떠한 task-specific 튜닝 없이도, 단순한 few-shot prompting만으로 시각적 task에 직접 활용 가능**하다.

### 기여 사항 (Contributions)

요약하면, 본 논문의 주요 기여는 다음과 같다:  
(i) 우리는 \*\*Flamingo 계열의 Visual Language Model (VLM)\*\*을 소개한다. 이 모델은 **few-shot input/output 예시만으로도 captioning, visual dialogue, visual question-answering**과 같은 다양한 **멀티모달 task**를 수행할 수 있다. **아키텍처적 혁신** 덕분에, Flamingo는 **텍스트와 시각 데이터를 임의로 섞은 입력을 효율적으로 수용하고, 자유형식의 텍스트를 생성**할 수 있다.  
(ii) 우리는 Flamingo 모델이 다양한 task에 대해 **few-shot learning을 통해 어떻게 적응 가능한지를 정량적으로 평가**한다. 특히, 우리는 이 접근법의 **디자인 결정이나 하이퍼파라미터 튜닝에 전혀 사용되지 않은 대규모의 held-out 벤치마크 세트**를 따로 보존하여, **편향되지 않은 few-shot 성능을 추정**하는 데 활용한다.  
(iii) Flamingo는 **언어 및 이미지/비디오 이해와 관련된 16개의 멀티모달 task에서 few-shot learning 기준으로 새로운 state of the art**을 달성한다. 이 중 **6개 task에서는 단 32개의 task-specific 예시만을 사용하고도, 기존 fine-tuned SOTA 모델보다 더 나은 성능**을 보여준다. 이는 **기존 SOTA보다 약 1000배 적은 task-specific training data**를 사용한 결과다. 또한, **더 큰 어노테이션 budget이 주어진다면**, Flamingo는 **VQAv2, VATEX, VizWiz, MSRVTTQA, HatefulMemes**와 같은 **5개의 추가적인 고난이도 벤치마크에서도 새로운 SOTA 성능**을 fine-tuning을 통해 달성할 수 있다.


## 2 Approach

이 섹션에서는 Flamingo를 설명한다. Flamingo는 **텍스트와 이미지/비디오가 섞인(interleaved) 입력을 받아 자유 형식의 텍스트를 출력하는 Visual Language Model**이다. Figure 3에 나타난 핵심 아키텍처 구성 요소들은 **사전학습된 vision 및 language model을 효과적으로 연결**하기 위해 설계되었다.  
첫째, **Perceiver Resampler**(Section 2.1)는 Vision Encoder로부터 얻은 시공간(spatio-temporal) feature들을 입력받아(입력은 이미지 또는 비디오일 수 있음), **고정된 개수의 visual token**을 출력한다.  
둘째, 이렇게 생성된 visual token들은 **사전학습된 Language Model(LM)** 내부에 **새롭게 초기화된 cross-attention layer**를 삽입하여 언어 생성에 조건(condition)으로 활용된다 (Section 2.2). 이 cross-attention layer는 **다음 token을 예측하는 과정에서 LM이 시각 정보를 유연하게 통합할 수 있도록** 해주는 강력한 구조이다.  
Flamingo는 **이미지/비디오와 섞인 텍스트 시퀀스가 주어졌을 때, 텍스트 $y$의 확률 분포**를 다음과 같이 모델링한다:

$$
p(y \mid x) = \prod_{\ell=1}^{L} p\left(y_{\ell} \mid y_{<\ell}, x_{\leq \ell}\right)
$$

여기서

* $y_{\ell}$은 입력 텍스트의 $\ell$-번째 language token,
* $y_{<\ell}$은 $\ell$-번째 token 이전의 모든 token,
* $x_{\leq \ell}$은 $y_{\ell}$ 이전에 등장한 이미지/비디오들의 집합,
* $p$는 Flamingo 모델에 의해 parameterized된 확률 분포이다.

이처럼 **텍스트와 시각 정보가 섞인 시퀀스를 처리할 수 있는 능력**(Section 2.3)은 Flamingo를 **GPT-3의 few-shot prompting과 유사하게 in-context few-shot learning에 자연스럽게 적용**할 수 있게 만든다. 본 모델은 다양한 dataset의 혼합으로 학습되며, 이에 대한 자세한 내용은 Section 2.4에서 설명한다.

![](https://cdn.mathpix.com/cropped/2025_07_26_7c316185968e7585aacbg-05.jpg?height=559&width=1385&top_left_y=249&top_left_x=370)

Figure 4: **GATED XATTN-DENSE layer.**  
**Language Model(LM)에 시각 정보를 조건으로 제공하기 위해**, 우리는 **기존의 사전학습된 고정된 LM layer 사이에 새로운 cross-attention layer를 삽입**한다. 이 cross-attention layer에서 **key와 value는 vision feature로부터 얻어지고**, **query는 language input으로부터 유도된다**.  
cross-attention 뒤에는 **dense feed-forward layer**가 이어진다.  
이러한 layer들은 **gate로 제어되며**, 이를 통해 **초기화 시점에서 LM의 본래 구조를 손상시키지 않고 안정성과 성능을 향상**시킬 수 있다.


### 2.1 Visual processing and the Perceiver Resampler

**Vision Encoder: 픽셀로부터 feature 추출**  
우리의 vision encoder는 사전학습된 **Normalizer-Free ResNet (NFNet)** \[10]이며, **F6 모델**을 사용한다. 이 vision encoder는 **이미지-텍스트 쌍으로 구성된 데이터셋**을 기반으로, **Radford et al. \[85]의 two-term contrastive loss**를 이용한 contrastive objective로 사전학습되었다.  
Encoder의 출력은 **2D 공간상의 feature grid**이며, 이는 \*\*1D 시퀀스로 평탄화(flatten)\*\*되어 사용된다.
비디오 입력의 경우, **초당 1프레임(FPS)으로 프레임을 샘플링**하고, 각 프레임은 개별적으로 인코딩된다. 이를 통해 **3D spatio-temporal feature grid**가 생성되며, 여기에 **학습된 temporal embedding**이 더해진다. 이렇게 얻은 feature는 **1D 시퀀스로 평탄화된 후 Perceiver Resampler에 입력**된다.  
contrastive 모델 학습 및 성능에 대한 자세한 내용은 각각 **Appendix B.1.3**과 **Appendix B.3.2**에서 설명되어 있다.


**Perceiver Resampler: 크기가 다양한 대형 feature map을 소수의 visual token으로 변환**  
이 모듈은 Figure 3에서 보여지듯, **vision encoder와 고정된(frozen) language model을 연결**하는 역할을 한다.
Perceiver Resampler는 **vision encoder로부터 추출된 이미지 또는 비디오 feature들을 입력으로 받아**, **고정된 개수(64개)의 visual output**을 생성한다. 이는 **vision-text cross-attention의 연산 복잡도**를 효과적으로 줄여준다.  
Perceiver \[48] 및 DETR \[13]에서와 유사하게, 우리는 **사전에 정의된 개수의 latent input query**를 학습하며, 이들은 Transformer에 입력되어 시각적 feature에 cross-attention을 수행한다.  
ablation study (Section 3.3)에서는 이러한 vision-language resampler 모듈이 **단순한 Transformer나 MLP보다 더 우수한 성능**을 보인다는 것을 입증한다.
이 모듈에 대한 시각적 예시, 아키텍처 세부사항, 그리고 pseudo-code는 **Appendix A.1.1**에 제시되어 있다.


### 2.2 Conditioning frozen language models on visual representations

**텍스트 생성(text generation)은 Perceiver Resampler가 생성한 시각 표현(visual representation)을 조건으로 하는 Transformer decoder에 의해 수행**된다. 우리는 **사전학습된 고정(frozen)된 텍스트 전용 LM 블록들과**, **Perceiver Resampler의 시각 출력을 cross-attend하는 방식으로 새로 학습되는 블록들을 교차(interleave)하여 배치**한다.

**Interleaving new GATED XATTN-DENSE layers within a frozen pretrained LM**  
우리는 \*\*사전학습된 LM 블록들은 그대로 고정(freeze)\*\*시키고, **새로 학습되는 gated cross-attention dense block**(Figure 4 참조)을 **기존 layer들 사이에 삽입**한다.
초기화 시, **조건부(conditioned) 모델의 출력이 기존 language model의 출력과 동일**하도록 하기 위해, 우리는 **tanh-gating mechanism**을 사용한다 \[41]. 이 방식에서는 **새로 삽입된 layer의 출력에 대해 $\tanh(\alpha)$를 곱한 후**, residual connection을 통해 들어온 입력에 더한다. 여기서 $\alpha$는 **layer마다 개별적으로 학습되는 scalar 파라미터**이며, 초기값은 0으로 설정된다 \[4].
따라서 초기 시점에서는 **모델의 출력이 기존 사전학습된 LM과 동일**하게 되며, **학습 안정성 및 최종 성능 향상**에 기여한다.  
ablation study (Section 3.3)에서는 **제안된 GATED XATTN-DENSE layer와 최근 대안들 \[22, 68]을 비교**하고, 이러한 layer를 **얼마나 자주 삽입할지를 조절함으로써 효율성과 표현력 간의 trade-off**를 탐구한다. 자세한 내용은 **Appendix A.1.2**를 참조하라.

**다양한 모델 크기 (Varying model sizes)**. 
우리는 **세 가지 모델 크기에서 실험을 수행**하였으며, 이는 각각 **Chinchilla 모델 \[42]의 1.4B, 7B, 70B 파라미터 버전**을 기반으로 한다. 이들을 각각 **Flamingo-3B**, **Flamingo-9B**, **Flamingo-80B**로 명명하였다. 본 논문 전체에서는 가장 큰 **Flamingo-80B**를 간단히 **Flamingo**라고 부른다.  
Frozen된 LM의 파라미터 수와 trainable한 vision-text GATED XATTN-DENSE 모듈의 규모는 모델 크기에 따라 증가시키되,
\*\*vision encoder(frozen)\*\*와 \*\*Perceiver Resampler(trainable)\*\*는 **모든 모델에서 동일한 크기로 고정**되어 있으며, 이는 전체 모델 크기에 비해 상대적으로 작다.  
자세한 구성은 **Appendix B.1.1**을 참조하라.


### 2.3 Multi-visual input support: per-image/video attention masking

Equation (1)에서 도입된 **image-causal modeling**은 전체 \*\*text-to-image cross-attention 행렬을 마스킹(masking)\*\*함으로써 구현되며, 이를 통해 **각 텍스트 토큰이 볼 수 있는 시각 토큰의 범위**를 제한한다.
즉, 주어진 텍스트 토큰에서 모델은 **interleaved 시퀀스 내에서 바로 직전에 등장한 이미지의 시각 토큰만을 attend**하며, 그 이전의 모든 이미지에는 직접적으로 attend하지 않는다 (이에 대한 공식화 및 도식은 Appendix A.1.3 참조).  
하지만 모델은 한 번에 하나의 이미지에만 직접 attend하더라도, \*\*LM 내부의 self-attention을 통해 간접적으로 모든 이전 이미지들과의 종속성(dependency)\*\*을 유지하게 된다.  
이러한 **single-image cross-attention 방식**은 중요한 장점이 있다.
즉, **훈련 시 사용된 이미지 수와 관계없이, 어떤 개수의 시각 입력에도 자연스럽게 일반화**할 수 있다는 점이다.
실제로 우리는 학습 시 interleaved dataset에서 **시퀀스당 최대 5장의 이미지**만을 사용했음에도, \*\*평가 시에는 이미지/비디오-텍스트 쌍(pair)\*\*을 **최대 32개까지 포함하는 시퀀스**에서도 성능 향상이 가능함을 확인했다.  
Section 3.3에서는 이 방식이, **모델이 이전 모든 이미지에 직접 cross-attend하도록 하는 방식보다 더 효과적**이라는 것을 실험적으로 보여준다.


### 2.4 Training on a mixture of vision and language datasets

우리는 Flamingo 모델을 **웹에서 수집한 세 가지 종류의 데이터셋 혼합물**로 학습시킨다:

* **웹페이지로부터 추출된 이미지-텍스트가 섞여 있는(interleaved) 데이터셋**,
* **이미지-텍스트 쌍 데이터셋**,
* **비디오-텍스트 쌍 데이터셋**.


**M3W: 이미지-텍스트가 섞인(interleaved) 데이터셋**  
Flamingo 모델의 few-shot 능력은 **텍스트와 이미지가 섞여 있는 데이터**로의 학습에 기반한다. 이를 위해 우리는 **MultiModal MassiveWeb (M3W)** 데이터셋을 구축하였다.
약 **4,300만 개의 웹페이지 HTML**로부터 텍스트와 이미지를 추출하였으며, **문서의 DOM(Document Object Model) 구조 내에서 텍스트와 이미지 요소의 상대적인 위치**를 기준으로 이미지의 텍스트 내 위치를 결정하였다.  
하나의 예시는 다음과 같이 구성된다:

* 페이지 내 이미지 위치에 `<image>` 태그를 일반 텍스트에 삽입하고,
* 각 이미지 앞과 문서 끝에는 학습 가능한 특수 토큰 `<EOC>`(end of chunk)를 삽입한다.

각 문서로부터 **임의로 256개의 토큰으로 구성된 subsequence를 샘플링**하고, **해당 시퀀스에 포함된 처음 5개 이미지**까지만 사용한다. **연산 비용 절약을 위해 그 이후의 이미지는 제거**한다. 자세한 내용은 Appendix A.3에 설명되어 있다.

**이미지/비디오-텍스트 쌍 데이터셋 (Pairs of image/video and text)**  
이미지-텍스트 쌍 데이터셋으로는 먼저 **ALIGN \[50] 데이터셋**을 활용하는데, 이는 약 **18억 개의 이미지와 alt-text 쌍**으로 구성되어 있다. 이와 보완적으로, 우리는 \*\*보다 긴 설명과 높은 품질을 목표로 하는 이미지-텍스트 쌍 데이터셋 LTIP (Long Text & Image Pairs)\*\*을 새롭게 수집하였고, 이는 **3억 1,200만 쌍**으로 구성된다.  
또한, \*\*정지 이미지 대신 비디오를 포함하는 데이터셋인 VTP (Video & Text Pairs)\*\*도 수집하였다.
VTP는 **평균 22초 분량의 짧은 비디오 2,700만 개**와 그에 대응되는 문장 단위의 설명 텍스트로 이루어져 있다.  
이러한 paired 데이터셋들은 **M3W와 문법(syntax)을 일치시키기 위해**, 각 caption 앞에는 `<image>`를 붙이고, 끝에는 `<EOC>`를 추가하였다 (자세한 내용은 Appendix A.3.3 참조).

**다중 목적 학습 및 최적화 전략 (Multi-objective training and optimisation strategy)**  
모델은 각 데이터셋에 대해 다음과 같은 \*\*시각 입력이 주어진 상태에서의 텍스트 생성 확률에 대한 expected negative log-likelihood의 가중합(weighted sum)\*\*을 최소화하도록 학습된다:

$$
\sum_{m=1}^{M} \lambda_{m} \cdot \mathbb{E}_{(x, y) \sim \mathcal{D}_{m}}\left[-\sum_{\ell=1}^{L} \log p\left(y_{\ell} \mid y_{<\ell}, x_{\leq \ell}\right)\right]
$$

여기서 $\mathcal{D}_m$은 $m$번째 데이터셋, $\lambda_m$은 해당 데이터셋의 가중치이다.  
**각 데이터셋별 가중치 $\lambda_m$를 조정하는 것이 성능에 핵심적인 요소**이다.
우리는 **모든 데이터셋에 걸쳐 gradient를 누적**하는 방식을 사용했으며, 이는 \[17]에서 제안된 **round-robin 방식보다 우수한 성능**을 보였다.
추가적인 학습 세부사항 및 ablation은 **Appendix B.1.2**에 수록되어 있다.


### 2.5 Task adaptation with few-shot in-context learning

Flamingo를 학습시킨 이후, 우리는 이를 **멀티모달 interleaved prompt를 조건으로 하여 다양한 시각적 task에 적용**한다.
우리는 **GPT-3 \[11]에서와 유사하게**, Flamingo 모델이 **in-context learning을 통해 새로운 task에 얼마나 빠르게 적응하는지**를 평가한다.
이를 위해 (image, text) 또는 (video, text) 형태의 **support example 쌍들을 interleave한 후**, 그 뒤에 \*\*쿼리 시각 입력(query visual input)\*\*을 추가하여 prompt를 구성한다 (자세한 구성은 Appendix A.2 참조).

* **open-ended 평가**는 beam search를 활용한 decoding으로 수행되고,
* **close-ended 평가**는 모델이 각 정답 후보에 대해 계산한 **log-likelihood 점수**를 이용해 수행된다.

또한 우리는 **zero-shot generalization**도 탐구하는데, 이때는 해당 task의 **텍스트 예시 2개만으로 prompt를 구성하고 시각 정보는 포함하지 않는다.**  
평가 시 사용한 하이퍼파라미터 및 추가적인 세부 사항은 **Appendix B.1.5**에 설명되어 있다.


| Method | FT | Shot | OKVQA (I) | VQAv2 (I) | COCO (I) | MSVDQA (V) | VATEX (V) | VizWiz (I) | Flick30K (I) | MSRVTTQA (V) | iVQA (V) | YouCook2 (V) | STAR (V) | VisDial (I) | TextVQA (I) | NextQA (I) | HatefulMemes (I) | RareAct (V) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Zero/Few shot SOTA | $x$ |  | [34] | [114] | [124] | [58] |  |  |  | [58] | [135] |  | [143] | [79] |  |  | [85] | [85] |
|  |  |  | 43.3 | 38.2 | 32.2 | 35.2 | - | - | - | 19.2 | 12.2 | - | 39.4 | 11.6 | - | - | 66.1 | 40.7 |
|  |  | (X) | (16) | (4) | (0) | (0) |  |  |  | (0) | (0) |  | (0) | (0) |  |  | (0) | (0) |
| Flamingo-3B | $x$ | 0 | 41.2 | 49.2 | 73.0 | 27.5 | 40.1 | 28.9 | 60.6 | 11.0 | 32.7 | 55.8 | 39.6 | 46.1 | 30.1 | 21.3 | 53.7 | 58.4 |
|  | $x$ | 4 | 43.3 | 53.2 | 85.0 | 33.0 | 50.0 | 34.0 | 72.0 | 14.9 | 35.7 | 64.6 | 41.3 | 47.3 | 32.7 | 22.4 | 53.6 | - |
|  | $x$ | 32 | 45.9 | 57.1 | 99.0 | 42.6 | 59.2 | 45.5 | 71.2 | 25.6 | 37.7 | 76.7 | 41.6 | 47.3 | 30.6 | 26.1 | 56.3 | - |
| Flamingo-9B | $x$ | 0 | 44.7 | 51.8 | 79.4 | 30.2 | 39.5 | 28.8 | 61.5 | 13.7 | 35.2 | 55.0 | 41.8 | 48.0 | 31.8 | 23.0 | 57.0 | 57.9 |
|  | $x$ | 4 | 49.3 | 56.3 | 93.1 | 36.2 | 51.7 | 34.9 | 72.6 | 18.2 | 37.7 | 70.8 | 42.8 | 50.4 | 33.6 | 24.7 | 62.7 | - |
|  | $x$ | 32 | 51.0 | 60.4 | 106.3 | 47.2 | 57.4 | 44.0 | 72.8 | 29.4 | 40.7 | 77.3 | 41.2 | 50.4 | 32.6 | 28.4 | 63.5 | - |
| Flamingo | $x$ | 0 | 50.6 | 56.3 | 84.3 | 35.6 | 46.7 | 31.6 | 67.2 | 17.4 | 40.7 | 60.1 | 39.7 | 52.0 | 35.0 | 26.7 | 46.4 | 60.8 |
|  | $x$ | 4 | 57.4 | 63.1 | 103.2 | 41.7 | 56.0 | 39.6 | 75.1 | 23.9 | 44.1 | 74.5 | 42.4 | 55.6 | 36.5 | 30.8 | 68.6 | - |
|  | $x$ | 32 | 57.8 | 67.6 | 113.8 | 52.3 | 65.1 | 49.8 | 75.4 | 31.0 | 45.3 | 86.8 | 42.2 | 55.6 | 37.9 | 33.5 | 70.0 | - |
|  |  |  | 54.4 | 80.2 | 143.3 | 47.9 | 76.3 | 57.2 | 67.4 | 46.8 | 35.4 | 138.7 | 36.7 | 75.2 | 54.7 | 25.2 | 79.1 |  |
| Pretrained FT SOTA | $\checkmark$ |  | [34] | [140] | [124] | [28] | [153] | [65] | [150] | [51] | [135] | [132] | [128] | [79] | [137] | [129] | [62] | - |
|  |  | (X) | ( 10 K ) | ( 444 K ) | ( 500 K ) | ( 27 K ) | ( 500 K ) | (20K) | ( 30 K ) | ( 130 K ) | (6K) | ( 10 K ) | (46K) | ( 123 K ) | (20K) | (38K) | (9K) |  |

**Table 1: 기존 state-of-the-art와의 비교**  
단일 Flamingo 모델은 다양한 이미지(I) 및 비디오(V) 이해 task에 대해 **few-shot learning만으로도 state-of-the-art 성능을 달성**하며, **기존의 zero-shot 및 few-shot 방법들을 단 4개의 예시만으로도 크게 능가**한다.
더 중요한 것은, **단 32개의 예시만 사용하고 모델 가중치를 전혀 업데이트하지 않은 상태로도**, Flamingo는 **수천 개의 annotated example로 fine-tuning된 기존 최고 성능의 방법들을 7개 task에서 능가**한다는 점이다.  
표에서 **가장 뛰어난 few-shot 성능은 굵게(bold)**, \*\*전체 최고 성능은 밑줄(underline)\*\*로 표시되어 있다.



## 3 Experiments

우리의 목표는 **다양하고 도전적인 task에 빠르게 적응할 수 있는 모델을 개발하는 것**이다. 이를 위해 우리는 **총 16개의 대표적인 멀티모달 이미지/비디오 및 언어 벤치마크**를 고려한다.  
프로젝트 진행 중 모델 설계 결정을 검증하기 위해, 이 중 **5개의 벤치마크는 개발용(DEV) 세트로 사용**되었다:
**COCO, OKVQA, VQAv2, MSVDQA, VATEX**.  
이러한 DEV 벤치마크에 대한 성능 추정치는 **모델 선택 과정에서의 편향이 존재할 수 있음**에 유의해야 한다. 이는 **유사한 벤치마크를 설계 검증 및 ablation에 활용한 기존 연구들에서도 동일하게 나타나는 현상**이다.  
이를 보완하기 위해 우리는 captioning, video question-answering, 그리고 visual dialogue 및 multi-choice question-answering과 같은 **잘 탐색되지 않은 영역을 포함한 추가적인 11개의 벤치마크**에서의 성능도 함께 보고한다.
이 평가용 벤치마크들에 대한 설명은 **Appendix B.1.4**에 제시되어 있다.
우리는 **모든 벤치마크에서 동일한 evaluation 하이퍼파라미터**를 사용하며, task에 따라 **총 4가지 few-shot prompt 템플릿 중 하나를 선택**해 적용한다 (자세한 내용은 **Appendix B.1.5** 참조).  
우리는 특히 강조한다: **이 11개의 평가용 벤치마크에서는 어떠한 설계 결정도 검증하지 않았으며**, **모델의 편향 없는 few-shot 학습 성능을 추정하는 목적**으로만 사용하였다.

보다 구체적으로 말하자면, 모델의 few-shot 학습 성능을 평가할 때는 **support 샘플들을 prompt로 주고, query 샘플에 대해 성능을 측정**한다.  
설계 결정 및 하이퍼파라미터 검증에 사용된 **DEV 벤치마크**에서는 **다음 4개의 subset을 사용**한다:

* validation support, validation query, test support, test query

반면, 그 외의 벤치마크에서는 **test support와 test query만 사용**하면 된다.  
이러한 subset 구성 방식은 **Appendix B.1.4**에 설명되어 있다.

**Section 3.1**에서는 Flamingo 모델의 few-shot 학습 성능을 보고하고, **Section 3.2**에서는 fine-tuning 결과를 제시하며, **Section 3.3**에서는 ablation study를 제공한다.  
추가적인 실험 결과는 **Appendix B.2**에 포함되어 있으며, 여기에는 **ImageNet 및 Kinetics700 분류 task**에서의 성능과 Flamingo의 contrastive 모델 성능이 포함된다.  
**Appendix C**에는 추가적인 qualitative 결과도 수록되어 있다.


### 3.1 Few-shot learning on vision-language tasks

**Few-shot 결과**  
결과는 **Table 1**에 제시되어 있다.  
Flamingo는 **16개의 벤치마크 전반에 걸쳐 기존의 모든 zero-shot 및 few-shot 방법들을 큰 차이로 능가**한다.  
이는 **task당 단 4개의 예시만으로 달성된 성과**로, **vision 모델이 새로운 task에 실용적이고 효율적으로 적응할 수 있음**을 보여준다.  
더 중요한 점은, **Flamingo가 수십만 개의 annotated example로 추가 fine-tuning된 state-of-the-art 방법들과도 종종 경쟁력 있는 성능을 보인다는 것**이다.  
심지어 **6개의 task에서는 Flamingo가 단 하나의 고정된 모델 가중치와 32개의 task-specific 예시만을 사용하고도 fine-tuned SotA를 능가**하는 성능을 기록했다.


| Method | VQAV2 test-dev test-std |  | COCO test | VATEX test | VizWiz test-dev test-std |  | MSRVTTQA test | VisDial valid test-std |  | YouCook2 valid | TextVQA valid test-std |  | HatefulMemes test seen |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ${ }^{3} 32$ shots | 67.6 | - | 113.8 | 65.1 | 49.8 | - | 31.0 | 56.8 | - | 86.8 | 36.0 | - | 70.0 |
| ${ }^{7}$ Fine-tuned | 82.0 | $\underline{82.1}$ | 138.1 | 84.2 | 65.7 | 65.4 | 47.4 | 61.8 | 59.7 | 118.6 | 57.1 | 54.1 | $\underline{86.6}$ |
| SotA | $81.3^{\dagger}$ [133] | $81.3^{\dagger}$ [133] | 149.6 $^{\dagger}$ [119] | $81.4^{\dagger}$ [153] | $57.2^{\dagger}$ [65] | $60.6^{\dagger}$ [65] | 46.8 [51] | 75.2 [79] | $\mathbf{7 5 . 4}^{\dagger}$ [123] | 138.7 [132] | 54.7 [137] | 73.7 [84] | $84.6^{\dagger}$ [152] |

**Table 2: Flamingo fine-tuning 시 SotA와의 비교**  
우리는 **few-shot learning만으로 SotA를 달성하지 못한 9개의 task**에 대해 **Flamingo를 fine-tuning**하였다.
그 결과, **그 중 5개의 task에서 Flamingo가 새로운 SotA를 달성**하였으며,
이는 **모델 앙상블, domain-specific metric 최적화 (예: CIDEr 최적화)** 등의 \*\*특수 기법을 사용하는 기존 방법들(† 표시)\*\*보다도 뛰어난 성능을 보인다.


|  | Ablated setting | Flamingo-3B original value | Changed value | Param. count $\downarrow$ | Step time $\downarrow$ | COCO CIDEr $\uparrow$ | OKVQA top1 $\uparrow$ | VQAv2 top1 $\uparrow$ | MSVDQA top1 $\uparrow$ | VATEX CIDEr $\uparrow$ | Overall score $\uparrow$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Flamingo-3B model |  |  |  | 3.2B | 1.74s | 86.5 | 42.1 | 55.8 | 36.3 | 53.4 | 70.7 |
| (i) | Training data | All data | w/o Video-Text pairs | 3.2B | 1.42s | 84.2 | 43.0 | 53.9 | 34.5 | 46.0 | 67.3 |
|  |  |  | w/o Image-Text pairs | 3.2B | 0.95s | 66.3 | 39.2 | 51.6 | 32.0 | 41.6 | 60.9 |
|  |  |  | Image-Text pairs $\rightarrow$ LAION | 3.2B | 1.74s | 79.5 | 41.4 | 53.5 | 33.9 | 47.6 | 66.4 |
|  |  |  | w/o M3W | 3.2B | 1.02 s | 54.1 | 36.5 | 52.7 | 31.4 | 23.5 | 53.4 |
| (ii) | Optimisation | Accumulation | Round Robin | 3.2B | 1.68s | 76.1 | 39.8 | 52.1 | 33.2 | 40.8 | 62.9 |
| (iii) | Tanh gating | $\checkmark$ | $\times$ | 3.2B | 1.74s | 78.4 | 40.5 | 52.9 | 35.9 | 47.5 | 66.5 |
| (iv) |  | GATED XATTN-DENSE | Vanilla Xattn | 2.4B | 1.16s | 80.6 | 41.5 | 53.4 | 32.9 | 50.7 | 66.9 |
|  | Cross-attention architecture |  | Grafting | 3.3 B | 1.74 s | 79.2 | 36.1 | 50.8 | 32.2 | 47.8 | 63.1 |
| (v) | Cross-attention frequency | Every | Single in middle | 2.0 B | 0.87s | 71.5 | 38.1 | 50.2 | 29.1 | 42.3 | 59.8 |
|  |  |  | Every 4th | 2.3 B | 1.02s | 82.3 | 42.7 | 55.1 | 34.6 | 50.8 | 68.8 |
|  |  |  | Every 2nd | 2.6 B | 1.24s | 83.7 | 41.0 | 55.8 | 34.5 | 49.7 | 68.2 |
| (vi) | Resampler | Perceiver | MLP | 3.2B | 1.85s | 78.6 | 42.2 | 54.7 | 35.2 | 44.7 | 66.6 |
|  |  |  | Transformer | 3.2 B | 1.81s | 83.2 | 41.7 | 55.6 | 31.5 | 48.3 | 66.7 |
| (vii) | Vision encoder | NFNet-F6 | CLIP ViT-L/14 | 3.1B | 1.58s | 76.5 | 41.6 | 53.4 | 33.2 | 44.5 | 64.9 |
|  |  |  | NFNet-F0 | 2.9 B | 1.45s | 73.8 | 40.5 | 52.8 | 31.1 | 42.9 | 62.7 |
| (viii) | Freezing LM | $\checkmark$ | $\boldsymbol{x}$ (random init) | 3.2B | 2.42s | 74.8 | 31.5 | 45.6 | 26.9 | 50.1 | 57.8 |
|  |  |  | $\boldsymbol{x}$ (pretrained) | 3.2 B | 2.42 s | 81.2 | 33.7 | 47.4 | 31.0 | 53.9 | 62.7 |

**Table 3: Ablation study 결과**  
각 행은 \*\*baseline Flamingo 실행 결과(맨 위 행)\*\*와 비교해야 한다.
여기서 **Step time**은 **모든 학습 데이터셋에 대해 gradient update를 수행하는 데 소요된 시간**을 나타낸다.


마지막으로, 우리는 설계 결정을 위해 **DEV 벤치마크만을 사용**했음에도 불구하고, **우리의 결과는 다른 벤치마크들에도 잘 일반화**되었으며, 이는 **우리 접근 방식의 범용성**을 입증한다.

**파라미터 수 및 shot 수에 따른 확장성(Scaling)**  
Figure 2에서 보여주듯이, **모델이 클수록 few-shot 성능이 더 우수**하며, 이는 GPT-3 \[11]와 유사한 경향이다. 또한, **shot 수가 많아질수록 성능도 향상**된다.  
우리는 특히, **가장 큰 모델이 더 많은 shot 수를 활용하는 데에 더 능숙**하다는 것을 발견했다.
흥미롭게도, Flamingo 모델은 **M3W에서 최대 5개의 이미지로 제한된 시퀀스로 학습되었음에도**, **추론 시에는 최대 32개의 이미지나 비디오로부터 성능 향상을 얻을 수 있다.**
이는 **Flamingo 아키텍처가 다양한 개수의 이미지나 비디오를 유연하게 처리할 수 있는 구조임을 보여주는 결과**이다.


### 3.2 Fine-tuning Flamingo as a pretrained vision-language model

비록 본 연구의 주요 초점은 아니지만, 우리는 **더 많은 데이터가 주어졌을 때 Flamingo 모델을 fine-tuning을 통해 특정 task에 적응시킬 수 있음**을 확인하였다.
**Table 2**에서는 **어노테이션 예산에 제한 없이** 주어진 task에 대해 **가장 큰 Flamingo 모델을 fine-tuning**하는 실험을 다룬다.  
구체적으로는, **짧은 학습 스케줄과 작은 learning rate를 사용해 모델을 fine-tuning**하며,
**더 높은 입력 해상도를 수용하기 위해 vision backbone도 함께 unfreeze**한다 (자세한 내용은 **Appendix B.2.2** 참조).  
그 결과, 이전에 제시한 **in-context few-shot 학습 성능을 능가하는 결과를 얻었으며**,
다음 **5개 task에서 새로운 state of the art**를 달성하였다:
**VQAv2, VATEX, VizWiz, MSRVTTQA, HatefulMemes**.


### 3.3 Ablation studies

**Table 3**에서는 **Flamingo-3B**를 사용해, **4-shot 설정에서 5개의 DEV 벤치마크의 validation subset**에 대해 수행한 ablation 실험 결과를 보고한다.
이 실험에서는 최종 모델과 비교해 **더 작은 batch size와 더 짧은 학습 스케줄**을 사용하였다.
**Overall 점수**는 각 벤치마크의 성능을 **Table 1의 SotA 성능으로 나눈 뒤 평균을 취해 계산**하였다.
추가적인 세부사항과 결과는 **Appendix B.3 및 Table 10**에 제시되어 있다.

**학습 데이터 구성의 중요성**  
(i)번 실험에서 보듯이, **적절한 학습 데이터 선택은 성능에 결정적인 영향을 미친다.**
interleaved 이미지-텍스트 데이터셋 **M3W를 제거하면 성능이 17% 이상 하락**하며, 전통적인 paired 이미지-텍스트 데이터셋을 제거해도 **성능이 9.8% 감소**한다.
이는 **서로 다른 유형의 데이터셋이 모두 중요함**을 보여준다.
또한, **비디오-텍스트 데이터셋을 제거하면 모든 비디오 관련 task에서 성능이 하락**하였다.  
우리는 또한 **자체 수집한 이미지-텍스트 쌍 데이터셋(ITP)을 공개된 LAION-400M \[96]으로 대체**하는 ablation도 수행했으며,
이 경우에도 **성능이 약간 저하됨**을 확인했다.  
(ii)번 실험에서는 **우리가 제안한 gradient accumulation 전략이 round-robin 업데이트 방식 \[17]보다 더 효과적임**을 보여준다.

**Frozen LM의 시각 조건화(Visual Conditioning)**  
(iii)번 실험에서는 **cross-attention 출력을 frozen LM에 병합할 때 사용하는 0으로 초기화된 tanh gating**의 효과를 확인한다.
이를 제거할 경우 **overall 점수가 4.2% 감소**하며, **학습 안정성도 악화됨**을 관찰했다.  
(iv)번 실험에서는 **다양한 조건화 아키텍처를 비교**했다.
**VANILLA XATTN**은 오리지널 Transformer decoder \[115]의 기본 cross-attention 구조이며,
\*\*GRAFTING \[68]\*\*은 **LM 자체는 그대로 사용하고**, 그 출력 위에 **새로운 self-attention 및 cross-attention layer 스택을 추가로 학습하는 방식**이다.  
이들에 비해 **GATED XATTN-DENSE 방식이 가장 우수한 성능을 보였다.**

**성능 대비 연산/메모리 효율 (Compute/Memory vs. Performance Trade-offs)**  
(v)번 실험에서는 **GATED XATTN-DENSE block을 삽입하는 빈도에 따른 trade-off**를 분석하였다.
모든 layer에 삽입하면 성능은 좋지만 **학습 시간이 증가**하며,
**4번째마다 삽입하는 경우, 학습 속도가 66% 빨라지면서도 overall 성능은 단 1.9%만 감소**했다.  
이러한 trade-off를 고려하여,
**Flamingo-9B에는 4번째마다**,
**Flamingo-80B에는 7번째마다 GATED XATTN-DENSE를 삽입**한다.  
(vi)번 실험에서는 **Perceiver Resampler를 MLP 또는 vanilla Transformer로 대체**했을 때를 비교하였다.
두 대안 모두 **성능이 더 낮고 속도도 느림**을 확인했다.

**Vision Encoder**  
(vii)번 실험에서는 **우리의 contrastive 학습된 NFNet-F6 vision encoder**와 공개된 **CLIP ViT-L/14 \[85] (224 해상도)**,
그리고 **더 작은 NFNet-F0**를 비교했다.  
그 결과, **NFNet-F6은 CLIP ViT-L/14보다 +5.8%, NFNet-F0보다 +8.0% 더 우수한 성능**을 보였다.
이는 **강력한 vision backbone의 중요성**을 강조한다.

**LM 고정(freezing)의 중요성: catastrophic forgetting 방지**  
(viii)번 실험에서는 **학습 시 LM layer를 고정시키는 것이 얼마나 중요한지**를 검증하였다.  
**LM을 scratch부터 학습시키면 성능이 12.9% 급감**하며, 사전학습된 LM을 **fine-tuning**하더라도 **성능이 8.0% 감소**한다.  
이는 학습 중 **모델이 사전학습에서 얻은 지식을 점점 잊어버리는 "catastrophic forgetting" \[71] 현상**이 발생함을 의미한다.
우리 실험에서는 **LM을 고정(freeze)시키는 것이, pretraining 데이터셋(MassiveText)을 혼합하여 함께 학습시키는 것보다도 더 효과적인 대안**이었다.

## 4 Related work

**Language modelling과 few-shot adaption**  
Transformer \[115]의 등장 이후, 언어 모델링은 상당한 발전을 이루어왔다. 대규모 데이터로 먼저 사전학습(pretraining)을 수행한 후, 다운스트림 task에 적응(adaptation)하는 패러다임은 이제 표준으로 자리 잡았다 \[11, 23, 32, 44, 52, 75, 87, 108]. 본 연구에서는 Flamingo의 기반 언어 모델로 70B 규모의 Chinchilla 모델 \[42]을 사용하였다. 몇몇 선행 연구에서는 언어 모델을 소수의 예시만으로 새로운 task에 적응시키는 다양한 방법들을 탐구해왔다. 예를 들어, 작은 adapter 모듈을 삽입하거나 \[43], LM의 일부만 fine-tuning 하거나 \[141], in-context 예시를 prompt에 삽입하거나 \[11], 또는 gradient descent를 통해 prompt 자체를 최적화하는 방식 \[56, 60] 등이 있다. 본 논문에서는 metric learning 기반 few-shot 학습 \[24, 103, 112, 117]이나 meta-learning 기반 접근법 \[6, 7, 27, 31, 91, 155]처럼 복잡한 방식이 아닌, GPT-3에서 소개된 in-context few-shot learning 기법 \[11]에서 영감을 받아 이를 Flamingo에 적용하였다.

**언어와 비전의 만남**  
이러한 언어 모델의 발전은 vision-language 모델링에도 큰 영향을 끼쳤다. 특히 BERT \[23]는 다수의 vision-language 관련 연구들 \[16, 28, 29, 38, 59, 61, 66, 101, 106, 107, 109, 118, 121, 142, 143, 151]에 영감을 주었다. 그러나 Flamingo는 이들과 달리 **새로운 task에 대해 fine-tuning을 요구하지 않는 점**에서 차별화된다. 또 다른 vision-language 모델 계열은 contrastive learning 기반 모델이다 \[2, 5, 49, 50, 57, 74, 82, 85, 138, 140, 146]. Flamingo는 이들과 달리 **텍스트를 생성할 수 있다는 점**에서 차별되며, Flamingo의 vision encoder는 이러한 contrastive 모델에 기반해 설계되었다. 본 연구와 유사하게, 일부 VLM은 autoregressive 방식으로 텍스트를 생성할 수 있도록 설계되어 있다 \[19, 25, 45, 67, 116]. 최근에는 여러 vision task들을 텍스트 생성 문제로 정식화하려는 연구도 진행되고 있다 \[17, 58, 119, 124, 154]. 사전학습된 대형 언어 모델을 기반으로 vision-language 모델을 구축하는 방향 역시 여러 연구들에서 탐색되고 있으며, 그중 일부 \[26, 68, 78, 114, 136, 144]는 \*\*catastrophic forgetting \[71]을 방지하기 위해 언어 모델 가중치를 고정(freeze)\*\*하는 방식을 제안한다. 우리도 이러한 접근을 따르며, **Chinchilla LM의 layer를 freeze**하고 그 내부에 **학습 가능한 layer들을 삽입**하였다. 그러나 기존 연구들과 달리, 우리는 **임의로 섞여 있는 이미지, 비디오, 텍스트를 모두 입력으로 수용할 수 있는 최초의 LM**을 제안한다는 점에서 차별성을 가진다.

**웹 규모의 비전-언어 학습 데이터셋**  
수작업으로 주석된 vision-language 데이터셋은 제작 비용이 매우 크기 때문에 규모가 상대적으로 작으며, 일반적으로 1만\~10만 개 수준이다 \[3, 15, 69, 122, 129, 139]. 이러한 데이터 부족 문제를 해결하기 위해, 여러 연구 \[14, 50, 98, 110]에서는 웹에서 쉽게 수집 가능한 이미지-텍스트 쌍 데이터를 자동으로 수집하는 방식이 제안되어 왔다. 본 연구는 이러한 paired 데이터 외에도, **이미지와 텍스트가 섞여 있는(multimodal interleaved) 전체 웹페이지를 하나의 시퀀스로 학습하는 것의 중요성**을 강조한다. 동시 진행된 연구인 CM3 \[1]에서는 웹페이지를 HTML 마크업으로 생성하는 방식을 택하였지만, 우리는 **텍스트 생성 문제를 단순화하기 위해 plain text만을 생성 대상으로 설정**하였다. 또한 우리는 **few-shot 학습 및 vision task 성능을 중점적으로 평가**한 반면, CM3 \[1]은 **zero-shot 또는 fine-tuning된 언어 전용 벤치마크**를 중심으로 평가를 진행하였다.


## 5 Discussion

**한계점**  
첫째, 우리의 모델은 사전학습된 언어 모델(LM)에 기반하고 있으며, 이로 인해 **해당 언어 모델의 약점을 그대로 물려받는다**는 부작용이 있다. 예를 들어, LM의 사전 지식(prior)은 일반적으로 유용하지만, **가끔 환각(hallucination)이나 근거 없는 추측**을 야기할 수 있다. 또한, LM은 **학습 시 사용된 시퀀스보다 긴 입력에 대해 일반화 성능이 낮으며**, **훈련 시 sample efficiency도 떨어지는 문제**를 가지고 있다. 이러한 문제들을 해결한다면, 본 분야의 발전을 가속화하고 Flamingo와 같은 VLM의 성능을 더욱 향상시킬 수 있을 것이다.

둘째, Flamingo는 **이미지 분류(classification) 성능에 있어서 최신 contrastive 학습 기반 모델들 \[82, 85]보다 뒤처진다.** 해당 contrastive 모델들은 **text-image retrieval**을 직접적으로 최적화하는데, 이는 분류 문제의 특수한 형태에 해당한다. 반면, Flamingo는 **보다 다양한 형태의 open-ended task를 다룰 수 있도록 설계되었다.** 이러한 두 방식의 장점을 결합하는 unified 접근법은 앞으로 중요한 연구 방향이 될 수 있다.

셋째, **in-context learning은 gradient 기반 few-shot 학습 방법에 비해 여러 장점**이 있지만, **적용 대상 task의 특성에 따라 단점도 존재**한다. 본 논문에서는 **수십 개 수준의 적은 예시만 주어졌을 때 in-context learning이 효과적임을 입증**하였다. 또한, in-context learning은 **추론(inference)만으로 동작하며, 일반적으로 하이퍼파라미터 튜닝 없이도 손쉬운 배포가 가능**하다는 이점이 있다. 그러나, **in-context learning은 demonstration 구성의 다양한 요소에 매우 민감**한 것으로 알려져 있으며 \[80, 148], **shot 수가 일정 수준 이상으로 늘어날 경우 계산 비용과 절대 성능이 비효율적으로 증가**한다. 따라서, **서로 보완적인 few-shot 학습 기법들을 결합하는 방법에 가능성**이 있다. 이와 관련된 한계점은 **Appendix D.1**에서 더 상세히 논의한다.

**사회적 영향**  
Flamingo는 여러 **긍정적인 잠재력**을 가지고 있지만, 동시에 **일정한 위험성**도 수반한다. Flamingo는 **적은 데이터로도 다양한 task에 빠르게 적응하는 능력**을 가지며, 이는 **비전문 사용자도 데이터 부족 환경에서 높은 성능을 달성할 수 있게 해준다.**
이러한 특성은 유익한 활용뿐 아니라 **악의적인 용도에도 악용될 수 있는 가능성**을 내포한다.  
Flamingo는 **기존 대형 언어 모델과 동일한 위험**, 예를 들어 **모욕적인 언어 생성, 사회적 편향 및 고정관념의 확산, 민감한 정보 누출** 등의 위험에 노출되어 있다 \[42, 126]. 더 나아가, Flamingo는 시각 입력을 처리할 수 있는 능력으로 인해, **입력 이미지의 내용에 따라 성별, 인종과 관련된 편향을 초래할 수 있는 위험성** 또한 내포하고 있다. 이러한 위험은 기존의 여러 시각 인식 시스템에서 관찰된 바 있다 \[12, 21, 37, 97, 147].  
우리는 **본 연구의 긍정적, 부정적 사회적 영향에 대한 보다 자세한 논의와 함께, 성별 및 인종 편향, 유해 출력에 대한 위험성의 조기 탐색 및 대응 전략**을 **Appendix D.2**에 정리해 두었다.
마지막으로, 이전의 언어 모델 연구들 \[72, 81, 111]에서 보여주었듯, Flamingo의 few-shot 능력은 이러한 위험을 완화하는 데에도 **긍정적인 역할을 할 수 있는 잠재력**이 있음을 언급한다.

**결론**  
우리는 본 논문에서 Flamingo를 제안하였다. Flamingo는 **최소한의 task-specific 학습 데이터만으로 이미지 및 비디오 task에 적용 가능한 범용 모델 계열**이다. 또한 우리는 Flamingo의 "대화"와 같은 **상호작용적 능력**을 정성적으로 탐구하였으며, 이는 **전통적인 비전 벤치마크를 넘어서는 유연성**을 보여준다.
우리의 결과는, **강력한 사전학습된 언어 모델과 시각 모델을 연결하는 것이 범용 시각 이해 모델로 가는 중요한 단계**임을 시사한다.

**감사의 말 및 연구 자금 지원 명시**  
본 연구는 **DeepMind**의 지원을 받아 수행되었다. 우리는 아래 동료들에게 유익한 논의, 제안, 피드백, 조언을 제공해 준 것에 감사드린다:
Samuel Albanie, Relja Arandjelović, Kareem Ayoub, Lorrayne Bennett, Adria Recasens Continente, Tom Eccles, Nando de Freitas, Sander Dieleman, Conor Durkan, Aleksa Gordić, Raia Hadsell, Will Hawkins, Lisa Anne Hendricks, Felix Hill, Jordan Hoffmann, Geoffrey Irving, Drew Jaegle, Koray Kavukcuoglu, Agustin Dal Lago, Mateusz Malinowski, Soňa Mokrá, Gaby Pearl, Toby Pohlen, Jack Rae, Laurent Sifre, Francis Song, Maria Tsimpoukelli, Gregory Wayne, 그리고 Boxi Wu.

## Checklist

1 **모든 저자에 대하여...**  
   (a) **초록 및 서론에서 제시한 주요 주장들이 논문의 기여 및 범위와 정확히 일치하나요?** \[예]  
   (b) **연구의 한계점을 기술했나요?** \[예] Section 5 참조.  
   (c) **연구의 잠재적인 부정적 사회적 영향에 대해 논의했나요?** \[예] 간략한 논의는 Section 5에, 전체 논의는 Appendix D.2에 포함되어 있음.  
   (d) **윤리 심사 가이드라인을 읽고, 논문이 이에 부합하는지 확인했나요?** \[예]

2 **이론적 결과를 포함하는 경우...**  
   (a) **모든 이론적 결과에 대한 전제 조건을 명확히 기술했나요?** \[해당 없음]  
   (b) **모든 이론적 결과에 대한 완전한 증명을 포함했나요?** \[해당 없음]

3 **실험을 수행한 경우...**  
   (a) **주요 실험 결과를 재현할 수 있도록 코드, 데이터, 사용 방법 등을 (부록이나 URL을 통해) 제공했나요?** \[아니오] 코드와 데이터는 독점적 자산임.  
   (b) **학습 세부사항(예: 데이터 분할, 하이퍼파라미터, 선택 방법 등)을 명시했나요?** \[예] Section 3 및 Appendix B 참조.  
   (c) **오차 막대(error bar)를 보고했나요? (예: 여러 번 실험 수행 후 random seed에 따른 변동성 등)** \[아니오] 실험 반복에 따른 분산이 크지 않았으며, 가장 큰 모델의 경우 계산 자원 제약으로 인해 여러 번의 학습은 현실적으로 어렵다고 판단함.  
   (d) **총 연산량 및 사용된 리소스 종류(GPU 종류, 내부 클러스터, 클라우드 제공업체 등)를 명시했나요?** \[예] 자세한 내용은 Appendix B.1.2에 있으며, 요약하면 **가장 큰 실험은 TPU 1536개로 15일간 학습됨.**

4 **기존 자산(코드, 데이터, 모델 등)을 사용하거나 새롭게 구축/공개한 경우...**  
   (a) **기존 자산을 사용했다면, 해당 제작자를 인용했나요?** \[예] 우리의 연구가 기반한 이전 방법과 적절한 경우 관련된 기존 데이터셋(예: ALIGN)을 적절히 인용함.  
   (b) **사용한 자산의 라이선스를 언급했나요?** \[해당 없음] 사용한 자산은 인용한 논문에서 나온 것이며, 논문 내 도표에 사용된 시각 자료의 라이선스는 Appendix G에서 언급함.  
   (c) **새로운 자산을 supplemental 자료나 URL을 통해 포함했나요?** \[아니오]  
   (d) **사용/수집한 데이터가 개인 정보 또는 타인의 데이터인 경우, 이에 대한 동의 여부를 논의했나요?** \[예] 우리의 데이터는 수백만 개의 웹페이지로부터 자동 수집된 것이며, 자세한 내용은 Appendix F의 Datasheets \[30]에 있음.  
   (e) **사용/수집한 데이터에 개인정보 또는 불쾌감을 줄 수 있는 콘텐츠가 포함되어 있는지 여부를 논의했나요?** \[예] Appendix F의 Datasheets \[30]에 기술되어 있음.

5 **크라우드소싱을 사용하거나 사람을 대상으로 한 연구를 수행한 경우...**  
   (a) **참가자에게 제공된 전체 지시사항과 스크린샷(해당되는 경우)을 포함했나요?** \[해당 없음]  
   (b) **참가자에게 발생할 수 있는 위험과 관련된 IRB 승인 여부 및 링크를 기술했나요?** \[해당 없음]  
   (c) **참가자에게 지급된 시간당 보상금과 총 보상액을 명시했나요?** \[해당 없음]


## Appendix

다음은 Appendix에 대한 개요이다.  
**Method (Appendix A)**
먼저 Appendix A.1에서는 모델에 대한 추가적인 세부 사항을 제공한다:

* **Perceiver Resampler**(Section 2.1에 설명됨)의 도식 및 pseudo-code는 **Appendix A.1.1**과 Figure 5에 수록되어 있다.
* **GATED XATTN-DENSE layer**(Section 2.2에 설명됨)의 유사한 도식은 **Appendix A.1.2**와 Figure 4에 포함되어 있다.
* **다중 이미지/비디오 attention 메커니즘**(Section 2.3)의 구현 세부 사항은 **Appendix A.1.3**에 설명되어 있다.
* 모든 모델 아키텍처에 대한 하이퍼파라미터는 **Appendix A.1.4**에 제시되어 있다.

이후, **in-context few-shot learning을 이용해 모델을 평가하는 방법**을 **Appendix A.2**에서 설명한다.
여기에는 few-shot prompt를 구성하는 방법, open-ended 및 close-ended task에 대한 예측을 수행하는 방식, zero-shot 수치 추정 방식, 그리고 더 많은 annotated 예시를 활용하기 위한 retrieval 및 ensembling 기법이 포함된다.

마지막으로, **Appendix A.3**에서는 학습 데이터셋에 대한 자세한 설명을 제공한다:

* **M3W 수집 과정**은 **Appendix A.3.1**에,
* **학습 중 M3W 샘플 처리 방식**은 **Appendix A.3.2**에,
* **LTIP 및 VTP 수집 과정**은 **Appendix A.3.3**에,
* **학습/평가 데이터셋 간 누수 방지를 위한 중복 제거 전략**은 **Appendix A.3.4**에 각각 기술되어 있다.

**Experiments (Appendix B)**  
먼저 **Appendix B.1**에서는 학습 및 평가 관련 세부사항을 추가로 설명한다:

* **Flamingo-3B, Flamingo-9B, Flamingo** 모델 구성은 **Appendix B.1.1**에,
* **학습 하이퍼파라미터**는 **Appendix B.1.2**에,
* **Contrastive 모델 사전학습 세부 정보**는 **Appendix B.1.3**에,
* **평가 벤치마크 및 데이터 분할**은 **Appendix B.1.4**에,
* **few-shot 학습 시의 하이퍼파라미터 설정**은 **Appendix B.1.5**에,
* **Figure 1 및 Figure 11의 정성적 대화 예시에서 사용된 대화 prompt**는 **Appendix B.1.6**에 각각 수록되어 있다.

다음으로, **Appendix B.2**에서는 Flamingo 모델의 추가 실험 결과를 제시한다.
여기에는 **분류 task에 대한 Flamingo 성능 (Appendix B.2.1)**, **fine-tuning 결과 (Appendix B.2.2)**, \*\*contrastive 모델의 zero-shot 결과 (Appendix B.2.3)\*\*가 포함된다.

마지막으로, **Appendix B.3**에서는 Flamingo 모델 (**Appendix B.3.1**)과 사전학습된 contrastive Visual Encoder (**Appendix B.3.2**)에 대한 추가적인 ablation study를 제공한다.

**Qualitative results (Appendix C)**  
**Appendix C**에서는 추가적인 정성적 결과들을 제시한다.
여기에는 **Figure 10 (단일 이미지 예시)**, **Figure 11 (대화 예시)**, \*\*Figure 12 (비디오 예시)\*\*가 포함된다.

**Discussion (Appendix D)**  
**Appendix D**에서는 본 연구의 **한계, 실패 사례, 광범위한 영향 및 사회적 영향**에 대해 보다 심도 있게 논의한다.

**Model card (Appendix E)**  
**Appendix E**에는 **Flamingo의 model card**가 포함되어 있다.

**Datasheets (Appendix F)**  
**Appendix F.1**에는 **M3W**, **Appendix F.2.1**에는 **LTIP**, **Appendix F.2.2**에는 **VTP**에 대한 datasheet가 각각 수록되어 있다.

**Credit for visual content (Appendix G)**  
논문에 사용된 모든 시각 자료에 대한 출처 및 저작권 표기는 **Appendix G**에 제공된다.

![](https://cdn.mathpix.com/cropped/2025_07_26_7c316185968e7585aacbg-23.jpg?height=790&width=678&top_left_y=242&top_left_x=371)

```
def perceiver_resampler(
    x_f, # The [T, S, d] visual features (T=time, S=space)
    time_embeddings, # The [T, 1, d] time pos embeddings.
    x, # R learned latents of shape [R, d]
    num_layers, # Number of layers
):
    """The Perceiver Resampler model."""
    # Add the time position embeddings and flatten.
    x_f = x_f + time_embeddings
    x_f = flatten(x_f) # [T, S, d] -> [T * S, d]
    # Apply the Perceiver Resampler layers.
    for i in range(num_layers):
        # Attention.
        x = x + attention_i(q=x, kv=concat([x_f, x]))
        # Feed forward.
        x = x + ffw_i(x)
    return x
```

**Figure 5**:
**Perceiver Resampler 모듈은 Vision Encoder로부터 출력된 가변 크기의 시공간(spatio-temporal) 시각 feature grid를 고정된 개수의 output token(그림에서는 5개)으로 매핑**한다.
이 과정은 **입력 이미지의 해상도나 입력 비디오 프레임 수와 무관하게 동작**한다.  
이 Transformer 구조에서는 **학습된 latent vector 집합이 query로 사용**되며,
\*\*key와 value는 시공간 시각 feature와 학습된 latent vector를 결합(concatenate)\*\*한 것으로 구성된다.


## A Method

## A. 1 Model details

## A.1.1 Perceiver Resampler

Section 2.1에서 간략히 설명한 내용을 확장하여, **Figure 5는 Perceiver Resampler가 비디오 예시를 처리하는 과정을 시각적으로 보여주며, 함께 pseudo-code도 제공**한다.
우리의 Perceiver Resampler는 **Jaegle et al. \[48]이 제안한 Perceiver 모델들과 유사한 철학**을 따른다.
우리는 **사전 정의된 개수의 latent input query를 학습하고**, 이를 \*\*평탄화(flatten)된 시각 feature $X_f$\*\*에 대해 cross-attention을 수행한다.  
이 시각 feature $X_f$는 다음과 같은 방식으로 생성된다.
비디오의 각 프레임(이미지의 경우 단일 프레임 비디오로 간주)의 feature에 대해 **학습된 temporal position encoding을 추가**한다.
주의할 점은, 우리는 **temporal encoding만 사용하고, 명시적인 spatial grid position encoding은 사용하지 않았다는 것**이다.
후자를 사용했을 때 **성능 향상이 관찰되지 않았기 때문**이다.  
그 배경에는 다음과 같은 이유가 있다:
**NFNet encoder와 같은 CNN은 채널 차원에서 암묵적으로 공간 정보를 내포하고 있는 것으로 알려져 있기 때문**이다 \[47].  
이후 시각 feature는 \*\*평탄화(flatten)되어 하나의 시퀀스로 연결(concatenate)\*\*되며, 이 과정은 Figure 5에 시각적으로 설명되어 있다.
**Perceiver Resampler의 출력 token 수는 학습된 latent query의 수와 동일**하다.  
DETR나 기존 Perceiver와는 달리,
우리는 \*\*학습된 latent로부터 계산한 key와 value를, 시각 feature $X_f$로부터 계산한 key와 value에 추가(concatenate)\*\*하는 방식으로 사용하며,
이 방식이 **약간 더 나은 성능**을 보이는 것을 확인하였다.


## A.1.2 GATED XATTN-DENSE details

우리는 **Figure 4에서 GATED XATTN-DENSE 블록의 구조와 그것이 frozen된 LM 블록에 어떻게 연결되는지**, 그리고 **해당 구조의 pseudo-code**를 함께 제공한다.

또한, **Figure 6에서는 Flamingo-3B 모델의 24개 LM layer에 대해, 학습 진행률(0%에서 100%)에 따른 tanh gating 값의 절대값 변화 추이**를 시각화하였다.
모든 frozen LM layer에서, **tanh gating 값의 절대값이 초기값인 0에서 빠르게 증가**하는 양상을 보이며,
이를 통해 각 layer가 **시각 정보를 적극적으로 활용하고 있음을 유추**할 수 있다.  
또한, **layer의 깊이에 따라 tanh gating 값의 절대값도 증가하는 경향이 관찰**되지만,
이는 **단정적인 결론을 내리기 어려운 현상**이다.
그 이유는 gating 이전의 activation의 scale 자체도 layer 깊이에 따라 달라질 수 있기 때문이다.

이러한 추가 layer들이 **최적화 동역학(optimization dynamics)** 및 **모델 자체에 미치는 영향**을 보다 깊이 이해하기 위해서는 **추가적인 후속 연구가 필요하다.**

![](https://cdn.mathpix.com/cropped/2025_07_26_7c316185968e7585aacbg-24.jpg?height=551&width=1378&top_left_y=245&top_left_x=371)

**Figure 6: Flamingo-3B의 서로 다른 layer에서 tanh gating의 절대값 변화 추이**
![](https://cdn.mathpix.com/cropped/2025_07_26_7c316185968e7585aacbg-24.jpg?height=337&width=1389&top_left_y=883&top_left_x=371)

**Figure 7: 섞여 있는(interleaved) 시각 데이터와 텍스트 지원 방식**  
웹페이지와 같이 이미지/비디오가 텍스트 사이에 섞여 있는 경우, 우리는 먼저 **텍스트 내 시각 데이터의 위치에 `<image>` 태그를 삽입**하고,
**시퀀스 시작을 나타내는 `<BOS>` 토큰**과 **chunk 종료를 나타내는 `<EOC>` 토큰**과 같은 **특수 토큰도 함께 삽입**한다.  
이미지들은 \*\*Vision Encoder와 Perceiver Resampler를 통해 개별적으로 처리되어 시각 토큰(visual token)\*\*으로 변환된다.
모델은 **각 텍스트 토큰에서, 그보다 앞서 등장한 마지막 이미지/비디오에 해당하는 시각 토큰만을 cross-attention**으로 참조한다.
$\phi$는 **각 텍스트 토큰이 참조할 수 있는 이미지/비디오를 나타내며**, 앞선 시각 정보가 없는 경우에는 0으로 표시된다.  
실제로 이러한 **선택적 cross-attention은 마스킹(masking)을 통해 구현**되며,
그 예시는 그림에서 **진한 파란색(마스킹 해제 / 볼 수 있음)**, \*\*연한 파란색(마스킹됨 / 볼 수 없음)\*\*으로 시각화되어 있다.


## A.1.3 Multi-visual input support

우리는 **Figure 7에서 특정 텍스트 토큰이 볼 수 있는 시각 토큰의 수를 제한하기 위해 사용하는 마스킹 방식**을 시각적으로 설명한다.
또한, **이미지/비디오와 텍스트가 섞인(interleaved) 시퀀스에 대한 표기법**도 공식화한다.

**시각 데이터와 텍스트가 섞인 시퀀스 (Interleaved sequences of visual data and text)**  
우리는 이미지/비디오와 텍스트가 섞여 있는 예시를 다룬다.
각 예시는 다음 세 가지로 구성된다:

* 텍스트 시퀀스 $y$,
* 이미지/비디오 시퀀스 $x$,
* 텍스트 내에서 이미지가 등장하는 위치에 대한 시퀀스.

시각 데이터의 위치를 기준으로, 우리는 다음과 같은 \*\*함수 $\phi: [1, L] \mapsto [0, N]$\*\*를 정의한다.
이 함수는 각 텍스트 위치 $\ell$에 대해, **그 위치 이전에 등장한 마지막 이미지/비디오의 인덱스를 반환**한다.
만약 그 위치 이전에 어떤 시각 데이터도 등장하지 않았다면 $\phi(\ell) = 0$이다.

이 함수 $\phi$는 \*\*Equation (1)에서 텍스트 토큰 $\ell$\*\*을 예측할 때 **어떤 시각 입력을 사용할 수 있는지를 정의**한다:

* 앞선 텍스트 토큰들의 집합은 $y_{<\ell} \triangleq (y_1, \ldots, y_{\ell-1})$,
* 앞선 이미지/비디오들의 집합은 $x_{\leq \ell} \triangleq \{x_i \mid i \leq \phi(\ell)\}$ 이다.


## A.1.4 Transformer architecture

Table 4에는 Flamingo 모델의 각 Transformer 구성 요소에 대해 다음 항목들을 정리하였다:
레이어 수 $L$, hidden dimension $D$, 헤드 수 $H$, 그리고 Feed-Forward(FFW)에서 사용하는 활성화 함수(Act.)이다.  
각 구성에서 key와 value의 차원은 $D / H$로 설정되며, **Perceiver Resampler**의 경우 96, **GATED XATTN-DENSE**와 **frozen LM**의 경우에는 128이다.
또한, 각 feed-forward MLP의 hidden dimension은 $4D$로 설정되어 있다.  
참고로, \*\*frozen LM은 GeLU activation \[39]\*\*으로 사전학습되었으며,
나머지 \*\*학습 가능한 Transformer layer들에는 Squared ReLU activation \[104]\*\*을 사용한다.
우리는 실험을 통해 **Squared ReLU가 GeLU보다 더 나은 성능**을 보인다는 사실을 확인하였다.


|  | Perceiver Resampler |  |  |  | GATED XATTN-DENSE |  |  |  | Frozen LM |  |  |  |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  | L | D | H | Act. | L | D | H | Act. | L | D | H | Act. |
| Flamingo-3B | 6 | 1536 | 16 | Sq. ReLU | 24 | 2048 | 16 | Sq. ReLU | 24 | 2048 | 16 | GeLU |
| Flamingo-9B | 6 | 1536 | 16 | Sq. ReLU | 10 | 4096 | 32 | Sq. ReLU | 40 | 4096 | 32 | GeLU |
| Flamingo | 6 | 1536 | 16 | Sq. ReLU | 12 | 8192 | 64 | Sq. ReLU | 80 | 8192 | 64 | GeLU |

**Table 4: Flamingo 모델의 Transformer에 대한 하이퍼파라미터**  
각 feedforward MLP의 hidden 크기는 $4D$이다.
$\mathbf{L}$: 레이어 수, $\mathbf{D}$: Transformer hidden 크기, $\mathbf{H}$: 헤드 수, Act.: Feed-Forward에서 사용하는 활성화 함수,
**Sq. ReLU**: Squared ReLU \[104].

![](https://cdn.mathpix.com/cropped/2025_07_26_7c316185968e7585aacbg-25.jpg?height=649&width=1403&top_left_y=678&top_left_x=361)

**Figure 8: Few-shot interleaved prompt 생성 방식**  
Flamingo가 예측을 수행해야 하는 쿼리와 함께, 몇 개의 task-specific few-shot 예시(즉, support example)가 주어졌을 때, 우리는 **이미지와 해당 텍스트를 번갈아(interleave) 배치하여 prompt를 구성**한다.  
이를 위해 **특정 포맷을 도입**하며,
vision-to-text task의 경우에는 **예상 응답 앞에 "Output:"을 붙이고**,
visual question-answering task의 경우에는
**"Question: {질문} Answer: {답변}"** 형식으로 prompt를 구성한다.

## A. 2 In-context few-shot evaluation details

**Flamingo 모델을 활용한 In-context learning**  
우리는 GPT-3 \[11]에서 사용된 접근 방식과 유사하게, **Flamingo 모델이 새로운 task에 빠르게 적응할 수 있는지 in-context learning을 통해 평가**한다.  
구체적으로, (image, text) 또는 (video, text) 형태의 **support example 집합**이 주어지며, 여기서 image 또는 video는 시각 입력이고, text는 \*\*예상되는 응답 또는 추가적인 task-specific 정보(예: 질문)\*\*이다.
또한, 모델이 예측을 수행해야 하는 **단일 visual query**도 함께 제공된다.  
이러한 정보를 바탕으로, 우리는 **Figure 8에서 보여주듯, support example들을 시각 쿼리 앞에 연결(concatenate)하여 멀티모달 prompt를 구성**한다.
별도의 명시가 없는 한, **example들의 연결 순서는 무작위로 선택**한다.

**Open-ended 및 Close-ended 평가 방식**  
**Open-ended 설정**에서는, 모델이 **쿼리 이미지 이후에 생성한 텍스트**를 해당 이미지에 대한 예측으로 간주하며, **첫 번째 `<EOC>`(end of chunk) 토큰이 생성될 때까지** 텍스트 생성을 진행한다.
특별한 언급이 없는 한, 우리는 항상 **beam size 3의 beam search**를 사용한다.  
**Close-ended 설정**에서는, 모든 후보 응답들을 쿼리 이미지 뒤에 **독립적으로 덧붙인 후**,
각 시퀀스에 대해 모델이 계산한 **log-likelihood를 기반으로 점수화**한다.
이 점수들을 이용해 \*\*후보 응답들을 신뢰도 순(높은 → 낮은)\*\*으로 정렬한다.

![](https://cdn.mathpix.com/cropped/2025_07_26_7c316185968e7585aacbg-26.jpg?height=277&width=1383&top_left_y=260&top_left_x=371)

**Figure 9: 학습 데이터셋**  
서로 다른 형식의 학습 데이터셋으로 구성된 혼합 구조를 나타낸다.
$N$은 하나의 예시에 포함된 시각 입력(이미지 또는 비디오)의 개수를 의미하며, \*\*paired image (또는 video)-text 데이터셋에서는 $N = 1$\*\*이다.
$T$는 비디오 프레임 수를 의미하며, \*\*이미지의 경우 $T = 1$\*\*이다.
$H, W, C$는 각각 \*\*높이(height), 너비(width), 색상 채널 수(channel)\*\*를 나타낸다.


**Zero-shot generalization**  
few-shot 예시가 없는 상황에서는, 모델이 inference 시 task에 적절한 자연어 설명을 조건으로 활용하도록 하는 **prompt engineering** 기법이 일반적으로 사용된다 \[85].
하지만 이러한 prompt를 검증하고 선택하는 과정은 성능에 큰 영향을 줄 수 있음에도 불구하고, **annotated 예시를 필요로 하므로 진정한 의미의 zero-shot으로 간주될 수 없다.**
게다가, **Perez et al. \[80]는 validation 과정에서 예시 수가 적을 경우, 성능이 쉽게 불안정해짐을 실험적으로 보였다.**  
우리의 연구에서는 zero-shot 성능을 평가하기 위해, **다운스트림 task에서 가져온 예시 중 해당 이미지 또는 비디오를 제거하여 텍스트만 포함된 2개의 예시로 prompt를 구성**하였다.
예를 들어, Figure 8 상단에 나오는 task의 경우, prompt는 다음과 같이 구성된다:
`<BOS>Output: This is a cat wearing sunglasses.<EOC>Output: Three elephants walking in the savanna.<EOC><image> Output:`
이때, 모델에는 **support 이미지가 제공되지 않는다.**  
우리는 텍스트 예시를 **1개만 제공하는 경우**, 모델이 그 예시와 유사한 형식의 응답을 생성하는 경향이 강해져 성능이 크게 저하됨을 확인했다.
반면, **2개를 제공하는 것이 실용성과 성능 간 균형 측면에서 가장 효과적**이었으며, **2개 이상을 제공하는 것은 성능을 약간만 향상시킬 뿐**이었다.
따라서 모든 zero-shot 결과에서는 **텍스트 예시 2개만을 사용**하였다.
실제로 이는 주어진 task에 대해 적절한 자연어 설명을 찾는 것보다 더 번거롭지 않다고 판단된다.
이러한 접근은 recent finding인 **demonstration의 구성 방식이 성능에 미치는 주요 요인**이라는 연구 \[76]와도 관련이 있다.  
**Close-ended task의 경우**, 후보 정답들에 대해 모델이 점수를 매기는 방식이므로, **zero-shot prompt에 단일 텍스트 예시조차 포함할 필요가 없다.**

**Retrieval-based In-Context Example Selection (RICES) \[136]**  
support set의 크기가 일정 수준을 넘어서게 되면, in-context learning으로 모든 예시를 효과적으로 활용하는 것이 어려워질 수 있다.
그 이유는 크게 두 가지다:
첫째, **모든 예시를 prompt에 넣기엔 너무 많은 연산 비용이 발생**하며,
둘째, **prompt의 길이가 학습 시 사용된 시퀀스 길이를 초과하면 일반화 성능이 저하될 위험**이 있기 때문이다 \[83].  
이런 경우, **prompt selection 기법을 사용하면 prompt 길이를 줄이고, 동시에 품질도 향상시켜 성능을 개선**할 수 있다 \[63].
우리는 이러한 접근 중 하나인 **Retrieval-based In-Context Example Selection (RICES)** 기법 \[136]을 따른다.  
구체적으로는, 쿼리 이미지가 주어졌을 때,
**사전학습된 frozen visual encoder로부터 추출한 시각 feature를 기준으로 support set 내에서 유사한 이미지들을 검색**한다.
그 후, **가장 유사한 상위 $N$개 예시를 연결하여 prompt를 구성**한다.  
또한, language model은 prompt에서 \*\*최근 등장한 정보에 민감(recency bias)\*\*하므로 \[148],
**유사도가 낮은 예시부터 높은 예시 순으로 정렬**하여 **가장 유사한 예시가 쿼리 바로 앞에 오도록 구성**한다.  
우리는 특히 **수백 개 이상의 class가 존재하는 분류 task 설정**에서 이 접근의 효과를 보여준다 (Appendix B.2.1 참조).
이 설정에서는 class마다 여러 개의 이미지/비디오가 주어지기 때문에, 예시 수가 prompt 길이를 초과하는 경우가 많다.

**Prompt ensembling**  
우리는 **close-ended setting에서 여러 개의 prompt에 대해 모델 출력을 앙상블하는 방법도 실험**하였다.
이 방식은 **RICES와도 결합 가능**하며, **유사한 예시들의 순서를 여러 가지로 섞어 생성한 prompt에 대해 모델 출력을 평균**하는 식으로 동작한다.  
구체적으로는, **선택된 few-shot 예시의 6가지 무작위 순열에 대해**,
각 정답 후보의 **log-likelihood를 모델이 계산한 뒤 평균을 취해 최종 점수를 산출**한다.


## A. 3 Training dataset details

우리는 **Figure 9에 시각적으로 나타나 있고, 아래에 설명된 다양한 데이터셋의 조합**을 **신중하게 선택하여 Flamingo 모델을 학습**시켰다.

## A.3.1 $M 3 W$ collection

\*\*$M3W$\*\*의 웹페이지 수집 및 크롤링 과정은 **MassiveWeb 데이터셋 \[86]을 수집할 때 사용된 방식과 유사한 절차**를 따른다.
우리는 먼저 **영어가 아닌 문서들을 필터링하여 제외**하고,
**이미지, 비디오, 텍스트에 걸쳐 explicit 콘텐츠를 식별하는 내부 필터를 통과하지 못한 문서들 또한 제거**한다.  
그 이후, 우리는 **텍스트와 이미지가 섞인 형태의 평문 콘텐츠**를 추출하기 위해 **커스텀 크롤러**를 사용하며,
이 과정은 Section 2.4에서 설명한 방식과 동일하다.  
$M3W$의 텍스트는 **MassiveWeb과 유사한 방식으로 수집**되지만,
우리는 여기에 추가로 **HTML 트리 상에서 동일한 수준에 위치한 이미지들도 함께 수집**한다.
스크래핑된 결과에서 **이미지가 포함되지 않은 문서들은 제거**한다.

그 다음, 우리는 **텍스트 품질을 높이기 위해 반복적이거나 품질이 낮은 문서를 제거**하고,
다음과 같은 조건에 해당하는 이미지를 제거하는 **이미지 필터링 절차**도 적용한다:

* 이미지의 너비 또는 높이가 64픽셀 미만인 경우
* 이미지의 가로세로 비율이 3 이상으로 지나치게 넓거나 좁은 경우
* 단색 이미지와 같이 **품질이 명확히 낮은 이미지**

이러한 필터링 이후에도 **이미지가 남아 있지 않은 문서들은 최종적으로 모두 폐기**한다.


## A.3.2 $M 3 W$ image-placement augmentation

Flamingo 모델을 평가할 때는 **이미지를 입력으로 제공하고 해당 이미지에 대한 텍스트를 생성하도록 프롬프트를 구성**한다.
이는 자연스럽게 **추론 시 이미지가 먼저 오고, 그에 따른 텍스트가 이어지는 순서**로 이어진다.

하지만, **interleaved된 M3W 데이터셋**(Section 2.4 참조)에서는 **이미지와 텍스트 간의 명확한 대응 관계가 일반적으로 알려져 있지 않으며**,
일부 경우에는 그 관계 자체가 **명확하게 정의되지 않을 수도 있다.**  
이를 설명하기 위한 동기 부여 예시로, 단순한 웹페이지 구조는 다음 두 가지 형태 중 하나일 수 있다:  
(a) This is my dog! <dog image> This is my cat! <cat image>  
(b) <dog image> That was my dog! <cat image> That was my cat!

이때 \*\*텍스트에 대응되는 이미지의 인덱스(index)\*\*는 이상적으로는 해당 텍스트와 **의미적으로 가장 관련이 깊은 이미지**를 가리키는 것이 바람직하다.
예를 들어 (a)에서는 다음 이미지가, (b)에서는 이전 이미지가 해당 텍스트와 관련 있을 것이다.  
그러나 **실제 웹페이지들에서는 이러한 의미적 일치를 일반적으로 판별하는 방법이 없기 때문에**, 우리는 다음과 같은 **단순화된 가정**을 적용한다:
**텍스트의 각 위치에서 가장 관련 있는 이미지는 직전에 등장한 이미지이거나 직후에 등장하는 이미지 중 하나**라는 것이다.
위 예시 구조들을 고려하여, 인덱스는 이러한 규칙에 따라 결정된다.

학습 과정에서는 각 웹페이지 샘플에 대해, 텍스트가 **이전 이미지와 대응되는지 또는 다음 이미지와 대응되는지를**
확률 $p_{\text{next}} = \frac{1}{2}$로 **무작위로 결정**한다.
이 방식은 불가피하게 (a)의 경우에서처럼 `"This is my cat!"`이라는 문장이 **강아지 이미지와 잘못 연결되는** 비자연적인 결과를 약 절반의 확률로 만들게 된다.  
Section 3.3에서 이 선택에 대한 ablation 실험을 수행한 결과,
$p_{\text{next}} = \frac{1}{2}$으로 설정하는 것이 **항상 이전 이미지(index=0)** 또는 **항상 다음 이미지(index=1)** 를 선택하는 것보다 **성능 면에서 약간의 이점을 가지는 것**으로 나타났다.
이는 이 무작위 선택 방식이 일종의 **"데이터 증강(data augmentation)" 효과를 제공**할 수 있음을 시사한다.


## A.3.3 LTIP and VTP: Visual data paired with text

우리의 interleaved image-text 데이터셋과 함께, 학습을 위해 여러 개의 paired vision-text 웹 데이터셋을 추가로 사용한다. 이 중 하나는 ALIGN \[50]으로, alt-text와 함께 제공되는 18억 개의 이미지로 구성되어 있다. ALIGN은 규모는 크지만 노이즈가 많고 이미지에 한정된다는 단점이 있다. 예를 들어, 이미지에 대한 alt-text 주석은 종종 부정확하거나 부실하다.  
이러한 한계를 보완하기 위해 두 가지 데이터셋을 추가로 사용한다:
LTIP (Long Text & Image Pairs) 데이터셋은 3억 1,200만 개의 이미지와 더 길고 풍부한 설명을 포함하며,
VTP (Video & Text Pairs) 데이터셋은 약 22초 분량의 짧은 영상 2,700만 개와 문장 수준의 캡션으로 구성되어 있다.
참고로, ALIGN 데이터셋에서 이미지 한 장당 평균 텍스트 길이는 12.4 토큰이지만, LTIP에서는 평균 20.5 토큰으로 더 자세한 설명을 포함한다.  
LTIP와 VTP는 고품질의 풍부한 이미지 설명을 제공하는 웹사이트 수십 개에서 수집되었으며, 이는 ALIGN보다 훨씬 적은 수의 웹사이트를 대상으로 한다.
이들 single-image 및 single-video 기반의 데이터셋은 앞서 설명한 M3W의 전처리 방식과 유사하게 전처리되었는데, 시퀀스 시작 부분(`<BOS>`) 바로 뒤에 `<image>` 태그를 삽입하고, 텍스트 뒤에는 `<EOC>` 토큰을 추가한 후 `<EOS>`로 마무리한다.  
또한 이 데이터셋들은 우리가 사용하는 모든 벤치마크(학습 및 평가 세트 모두)에 대해 중복을 제거하였다. 중복 제거는 Appendix A.3.4에서 설명된 바와 같이 이미지 유사도를 기반으로 수행되었다.
LTIP와 VTP에 대한 데이터시트는 각각 Appendix F.2.1 및 Appendix F.2.2에 수록되어 있다.


|  | Requires model sharding | Frozen |  | Trainable |  | Total count |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  |  | Language | Vision | GATED XATTN-DENSE | Resampler |  |
| Flamingo-3B | $\times$ | 1.4 B | 435 M | 1.2 B (every) | 194 M | 3.2B |
| Flamingo-9B | $\times$ | 7.1 B | 435 M | 1.6B (every 4th) | 194 M | 9.3B |
| Flamingo | $\checkmark$ | 70 B | 435 M | 10B (every 7th) | 194 M | 80B |

**표 5: Flamingo 모델의 파라미터 수**  
우리는 frozen된 language model(LM)과 학습 가능한 vision-text GATED XATTN-DENSE 모듈의 파라미터 수를 증가시키는 데 중점을 두었으며, frozen vision encoder와 학습 가능한 Resampler는 모든 모델에서 고정된 작고 일정한 크기로 유지하였다.
표에서 괄호 안의 수치는 GATED XATTN-DENSE가 원래 language model 블록에 삽입되는 빈도수를 나타낸다.


## A.3.4 Dataset deduplication against evaluation tasks

우리는 내부 deduplication 도구를 사용하여 학습 데이터셋과 평가 데이터셋 간의 중복 제거를 수행했다. 이 deduplication 파이프라인은 잠재적 중복일 가능성이 있는 이미지들의 임베딩을 서로 가깝게 매핑하도록 학습된 visual encoder를 기반으로 한다. 이미지 임베딩이 계산된 후, 학습 이미지에 대해 빠른 근사 최근접 이웃 탐색을 수행하여 평가 데이터셋으로부터 중복 후보를 검색한다.  
paired image-text 데이터셋에 대해서는, LTIP 및 ALIGN 학습 이미지들을 다음과 비교하여 deduplication을 수행했다: ImageNet (train, val), COCO (train, valid, test), OK-VQA (train, valid, test), VQAv2 (train, valid, test), Flickr30k (train, valid, test), VisDial (train, valid, test).

우리는 VizWiz, HatefulMemes, TextVQA 데이터셋과는 이미지 중복 제거를 수행하지 않았는데, 이는 Flamingo 모델의 학습이 완료된 이후에 해당 평가들을 진행했기 때문이다. 그러나 이러한 결정이 결과에 영향을 주지는 않았다고 판단한다. 그 이유는 이들 데이터셋의 이미지는 웹에서 수집된 것이 아닐 가능성이 높기 때문이다. VizWiz의 이미지는 특정 모바일 앱을 통해 수집되어서 다운로드를 통해서만 얻을 수 있고, HatefulMemes는 연구자들이 생성한 밈이며, TextVQA는 OpenImages로부터 가져온 이미지이기 때문이다.

한편, $M3W$ 데이터셋에 대해서는 deduplication을 수행하지 않았다. 그 이유는 해당 데이터셋의 하나의 학습 샘플이 여러 개의 이미지를 포함하는 전체 웹페이지이기 때문에, 벤치마크에 사용되는 이미지들과 중복될 가능성이 낮다고 보았기 때문이다. 이 가설을 검증하기 위해 우리는 $M3W$의 1억 8,500만 개의 개별 이미지에 대해 중복 탐지를 수행했으며, 그 결과는 다음과 같다: ImageNet, COCO, OK-VQA, VQAv2, Flickr30k, VisDial의 validation 및 test split으로부터 총 1,314개의 잠재적 중복 후보가 발견되었으며, 그 중 정확히 일치하는 중복은 단 125개뿐이었다.

비디오 데이터셋에 대해서는 VTP (2,700만 개의 비디오)에 대해 별도의 중복 제거를 수행하지 않았다. 이는 우리가 수집한 VTP 비디오가 YouTube나 Flickr에서 가져온 것이 아니며, 반면 우리가 사용한 모든 비디오 평가 데이터셋은 인터넷에서 수집된 YouTube 혹은 Flickr 기반의 것이기 때문이다.


## B Experiments

## B. 1 Training and evaluation details

## B.1.1 Models

우리는 세 가지 모델 크기에 걸쳐 실험을 수행하였으며, 고정된 language model의 크기를 1.4B에서 7B, 70B로 확장하면서, 그에 따라 다른 구성 요소들의 파라미터 수도 조정하였다. 모든 실험에서 vision encoder는 고정된 pretrained 모델을 사용하였고, 별도로 명시된 경우를 제외하면 contrastive 방식으로 학습된 NFNet-F6 모델을 사용하였다(자세한 내용은 Appendix B.1.3 참조). Perceiver Resampler는 모든 모델 크기에서 약 2억 개의 파라미터를 갖도록 유지하였다.

GATED XATTN-DENSE layer를 얼마나 자주 삽입할 것인가는 메모리 제약과 downstream 성능 간의 trade-off에 의해 결정된다. 우리는 이 trade-off의 최적 지점을 작은 규모의 모델에서 식별한 후, 그 결과를 대규모 모델 아키텍처로 전이시켰다.

최종적으로 우리는 다음과 같은 세 가지 모델을 구성하였다: Flamingo-3B, Flamingo-9B, 그리고 Flamingo-80B.

* **Flamingo-3B**는 \[42]의 1.4B 크기의 frozen language model 위에 구축되었으며, 각 Transformer block 앞에 visual input을 참조하는 GATED XATTN-DENSE layer를 추가하였다. 이로 인해 약 14억 개의 추가 학습 파라미터가 발생한다.

- **Flamingo-9B**는 \[42]의 7B frozen language model 위에 구축되었으며, 가장 첫 번째 layer부터 시작하여 매 4번째 Transformer block 앞에 GATED XATTN-DENSE layer를 추가하였다. 이로 인해 약 18억 개의 추가 학습 파라미터가 발생한다.

* **Flamingo-80B**는 고정된 Chinchilla 70B language model \[42] 위에 구축되었으며, 가장 첫 번째 layer부터 시작하여 매 7번째 Transformer block 앞에 GATED XATTN-DENSE layer를 추가하였다. 이로 인해 약 100억 개의 추가 학습 파라미터가 발생한다. 논문 전반에서는 이 모델을 간단히 **Flamingo**로 지칭한다.

Table 5에는 각 모델 구성 요소의 파라미터 수와 모델 분산 학습(sharding) 요구사항이 정리되어 있으며, Transformer 아키텍처에 대한 보다 구체적인 세부 사항은 Appendix A.1.4에 제공되어 있다. 또한, Flamingo 모델 카드 \[77]는 Appendix E에 포함되어 있다.

## B.1.2 Training details for the Flamingo models

**데이터 증강 및 전처리**
실험적으로 우리는 paired dataset의 텍스트 샘플 앞에 확률 0.5로 공백 문자 하나를 무작위로 추가하는 것이 효과적이라는 것을 발견했다. 이는 subword tokenizer가 단어 앞에 공백이 있느냐 없느냐에 따라 서로 다른 토큰으로 매핑한다는 사실에서 기인한다. 이러한 방식은 tokenizer의 이 인공적인 특성(tokenizer artifact)에 대한 불변성을 학습시킬 수 있도록 도와주며, 많은 샘플에서 이미 문장 부호가 부족한 점을 고려했을 때 문장의 정확도에 크게 영향을 주지 않는다. 이 처리는 다양한 태스크 전반에서 성능 향상을 이끌어냈다.

시각적 입력은 해상도 $320 \times 320$으로 조정되며, 종횡비는 유지된다. 필요한 경우, 평균값으로 패딩이 추가된다. 이 해상도는 Vision Encoder의 contrastive pretraining에서 사용된 $288 \times 288$보다 더 높은 해상도이다 (Appendix B.1.3 참고). 최종 학습 단계에서 해상도를 높인 이유는, CNN 기반 모델은 테스트 시점에 더 높은 해상도를 사용하면 성능이 향상될 수 있다는 기존 연구 \[113]에 기반한다. 이는 frozen Vision Encoder에 대해 backpropagation을 수행하지 않기 때문에 계산량 및 메모리 비용을 적당한 수준으로 유지하면서 성능 향상을 얻을 수 있다. 또한, 우리는 random left/right flip과 색상 변화(color augmentation)도 적용한다.

Interleaved dataset (Section 2.4 참조)의 경우, $M3W$에서 샘플을 선택할 때 선택된 이미지 인덱스 $\phi$를 확률 $p\_{\text{next}}$로 가볍게 무작위화하여 증강한다. 이 증강 기법은 Appendix A.3.2에 자세히 설명되어 있으며, 우리가 선택한 $p\_{\text{next}} = \frac{1}{2}$ 값은 Appendix B.3.1에서 ablation 실험을 통해 분석되었다. 비디오 학습에서는 각 학습 비디오에서 1초 간격으로 8프레임을 샘플링하여 사용한다. 모델은 학습 시 8개의 고정된 프레임을 사용했지만, 추론 시에는 3 FPS로 30프레임을 입력으로 넣는다. 이는 Perceiver Resampler의 학습된 temporal position embedding을 선형 보간하여 구현된다.

**손실 함수 및 최적화**
모든 모델은 AdamW optimizer를 사용해 학습되었고, gradient의 global norm은 1로 clipping되며, Perceiver Resampler에 대해서는 weight decay를 적용하지 않고, 나머지 학습 가능한 파라미터에는 0.1의 weight decay를 사용했다. learning rate는 처음 5,000 step 동안 선형적으로 $10^{-4}$까지 증가시키고 이후에는 일정하게 유지된다 (학습률을 감소시켜도 성능 개선은 관찰되지 않음). 별도로 명시되지 않는 한, 모델은 총 50만 step 동안 학습된다. 학습에는 네 개의 데이터셋인 $M3W$, ALIGN, LTIP, VTP가 사용되며, 각 데이터셋의 가중치 $\lambda\_m$는 각각 $1.0, 0.2, 0.2, 0.03$이다. 이 가중치는 소규모 모델에서의 실험을 통해 결정되었고, 이후 모든 실험에서 동일하게 유지되었다. 배치 크기는 실험 조건에 따라 달라지며, 이는 이후 섹션에서 명시된다.

**인프라 및 구현**
모델 및 관련 인프라는 JAX \[8]와 Haiku \[40]를 사용해 구현되었다. 학습 및 평가 과정은 TPUv4 인스턴스에서 수행되었다. 가장 큰 모델(파라미터 수 800억)은 총 1,536개의 TPU 칩에서 15일간 학습되었으며, 16개의 디바이스에 걸쳐 sharding되었다. Embedding, Self-Attention, Cross-Attention, FFW layer들은 모두 16-way Megatron 방식 \[99]으로 모델 병렬화되었으며, Vision encoder인 NFNet은 병렬화되지 않았다. optimizer state는 ZeRO stage 1 \[88]을 이용해 sharding되었다. 모든 학습 가능한 파라미터와 optimizer 누적값은 float32로 저장 및 업데이트되며, activation과 gradient는 파라미터를 float32에서 bfloat16으로 변환한 후 bfloat16으로 계산된다. frozen 파라미터는 bfloat16 형식으로 저장 및 적용된다.


## B.1.3 Contrastive model details

**Vision encoder는 language encoder와 함께 처음부터 학습되며**, 이 두 encoder를 사용해 이미지와 텍스트 쌍을 각각 인코딩한 뒤, 공통 임베딩 공간으로 투영하고 L2 normalization을 수행한다. 이렇게 얻어진 임베딩을 바탕으로, 정답 쌍의 임베딩 간 유사도는 최대화하고, 잘못된 쌍의 유사도는 최소화하도록 학습한다. 이를 위해 **multi-class cross-entropy loss**를 사용하며, 정답 쌍의 이미지-텍스트는 positive example로 간주되고, 같은 배치 내의 나머지 쌍들은 negative example로 처리된다. 이 loss는 CLIP \[85]에서 사용한 것과 동일하며, text-to-image와 image-to-text 방향의 두 개의 contrastive loss로 구성되어 있다. 최종 log-softmax 레이어에는 학습 가능한 temperature 파라미터를 사용한다 \[9].

text-to-image loss는 다음과 같이 정의된다:

$$
L_{\text {contrastive:txt } 2 i m}=-\frac{1}{N} \sum_{i}^{N} \log \left(\frac{\exp \left(L_{i}^{\top} V_{i} \beta\right)}{\sum_{j}^{N} \exp \left(L_{i}^{\top} V_{j} \beta\right)}\right)
$$

image-to-text loss는 이와 유사하게 다음과 같다:

$$
L_{\text {contrastive:im } 2 \text { txt }}=-\frac{1}{N} \sum_{i}^{N} \log \left(\frac{\exp \left(V_{i}^{\top} L_{i} \beta\right)}{\sum_{j}^{N} \exp \left(V_{i}^{\top} L_{j} \beta\right)}\right)
$$

이 두 loss의 합을 최소화하도록 학습된다. 여기서 $V\_{i}$와 $L\_{i}$는 각각 vision과 language encoder가 생성한 $i$번째 샘플의 normalized embedding을 의미하며, $\beta$는 학습 가능한 inverse temperature 파라미터이고, $N$은 배치 크기이다. **language encoder로는 BERT \[23] 아키텍처**를 사용한다. language encoder와 vision encoder의 출력은 각각 token 단위와 spatial 위치 단위로 평균을 내는 **mean pooling**을 거친 후, 공통 임베딩 공간으로 투영된다. Flamingo 본 모델에는 이 contrastive 방식으로 학습된 **vision encoder만** 사용된다.

**Vision encoder는 ALIGN과 LTIP 데이터셋**으로 사전학습되며, 학습 이미지 해상도는 $288 \times 288$이다. 공통 임베딩 공간의 크기는 1376이고, 배치 사이즈는 16,384이다. 총 120만 step 동안 학습되며, 각 step은 2회의 gradient 계산으로 구성된다. 학습은 **512개의 TPUv4 칩** 위에서 이루어졌고, learning rate는 $10^{-3}$에서 선형적으로 0까지 감소시킨다. 학습 중에는 이미지에 대해 색상 변화(color augmentation)와 수평 뒤집기(random horizontal flip)를 적용한다. tokenizer는 Jia et al. \[50]에서 사용한 것을 따른다. optimizer로는 **Adam**을 사용하며, **label smoothing 0.1**을 적용한다. NFNet encoder에는 **$10^{-2}$의 adaptive gradient clipping (AGC)** \[10]을, BERT encoder에는 **global norm gradient clipping 10**을 적용한다.

사전학습된 모델의 성능을 평가하기 위해 **zero-shot image classification과 retrieval**을 추적한다. zero-shot classification의 경우, 이미지와 클래스 이름 간의 **image-text retrieval**을 수행한다. Radford et al. \[85]를 따라, `"A photo of a {class_name}"`과 같은 템플릿을 사용해 여러 개의 문장을 임베딩한 후 평균을 취하는 **prompt ensembling** 기법을 사용한다.