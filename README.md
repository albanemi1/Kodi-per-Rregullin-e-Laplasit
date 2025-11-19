# Laplace’s Rule of Succession – Shembulli i rating-eve në Amazon (Projekt në Python)

---

## 1. Qëllimi i projektit

Qëllimi i këtij projekti është që, duke përdorur **Rregullin e Laplace-it (Laplace’s rule of succession)** në kuadër të probabilitetit bayesian, të vendosim:

> Nga cili ofrues në Amazon është më e arsyeshme të blihet produkti kur çmimi është pothuajse i njëjtë, por ndryshojnë **rating-et** (pozitiv / negativ) dhe **numri i vlerësimeve**?

Hapat kryesorë:

1. Përshkrimi i saktë i problemit.
2. Ndërtimi i një modeli probabilistik (Bernoulli + Beta).
3. Derivimi i formulas së Laplas-it.
4. Zbatimi i formulës në **Python**.
5. Analiza e detajuar e kodit, e ndarë në **blloqe**.

Ky README është i dizajnuar në mënyrë që të mund të lexohet edhe nga dikush që nuk ka shumë njohuri paraprake në probabilitet ose Python.

---

## 2. Përshkrimi i problemit (Detyra)

Amazon sugjeron **tre ofrues** të një produkti:

- Çmimet: praktikisht të njëjta.
- Vendimi duhet të merret vetëm nga **vlerësimet e klientëve**.

Të dhënat:

- **Ofruesi 1** – 10 vlerësime, 100% pozitive  
- **Ofruesi 2** – 50 vlerësime, 96% pozitive  
- **Ofruesi 3** – 200 vlerësime, 93% pozitive  

Detyra:

> Nga cili ofrues duhet blerë produkti, nëse synohet ai me **probabilitetin më të madh** që review-i i ardhshëm të jetë pozitiv?

Këtu nuk mjafton vetëm përqindja naive. Duhet të merret parasysh edhe **numri i vlerësimeve**. Kjo bëhet në mënyrë elegante me **Laplace’s Rule of Succession**.

---

## 3. Qasja naive: frekuenca $s/n$

Qasja më e thjeshtë për të vlerësuar probabilitetin e review-it pozitiv është:

$$\{p} = \frac{s}{n}$$

ku:

- $s$ = numri i review-eve **pozitive**  
- $n$ = numri i review-eve **totale**

Për ofruesit:

1. Ofruesi 1:  

$$
p_1 = \frac{10}{10} = 1.00 = 100\mathrm{\\%}
$$


2. Ofruesi 2:  

$$
p_2 = \frac{48}{50} = 0.96 = 96\mathrm{\\%}
$$  

3. Ofruesi 3:  

$$
p_3 = \frac{186}{200} = 0.93 = 93\mathrm{\\%}
$$



Sipas kësaj logjike naive, do të zgjidhnim **Ofruesin 1** (100\%).

Problemi:

- Ofruesi 1 ka vetëm **10 vlerësime** → shumë pak të dhëna.
- Mjafton vetëm një review negativ dhe shkalla bie në 90\%.
- Ofruesi 3 ka **200 vlerësime** → shumë më shumë informata, megjithëse përqindja është më e ulët.

Pra, qasja naive nuk e dallon qartë ndryshimin ndërmjet:

- **përqindjes së suksesit** ($s/n$),
- dhe **volumit të të dhënave** ($n$).

Duhet një qasje më e kujdesshme – këtu futen **Bayes** dhe **Laplace**.


---

## 4. Modeli probabilistik bayesian

### 4.1. Modelimi i review-eve (Bernoulli)

Supozimet:

- review pozitiv → e modelojmë si **1**,  
- review negativ → e modelojmë si **0**.

Secili review është eksperiment **Bernoulli** me probabilitet:

$$
P(\text{review pozitiv}) = \theta, \quad
P(\text{review negativ}) = 1 - \theta
$$

Për secilin ofrues kemi një vlerë të ndryshme të $\theta$:

- $\theta_1$ – probabiliteti i review-ve pozitive për Ofruesin 1  
- $\theta_2$ – probabiliteti i review-ve pozitive për Ofruesin 2  
- $\theta_3$ – probabiliteti i review-ve pozitive për Ofruesin 3  

Nëse për një ofrues kemi:

- $s$ review pozitive  
- $f$ review negative  
- $n = s + f$ review gjithsej  

atëherë **likelihood** i të dhënave (probabiliteti i të dhënave duke supozuar se $\theta$ është e dhënë) është:

$$
P(\text{të dhënat} \mid \theta) = \theta^s (1 - \theta)^f
$$

---


### 4.2. Prior – çfarë dimë para të dhënave?

Para se të vlerësohen review-et, supozojmë që:

- nuk kemi arsye të dyshojmë që produkti është shumë i mirë apo shumë i keq,
- të gjitha vlerat e $\theta$ në intervalin $[0, 1]$ janë **po aq të mundshme**.

Kjo përshkruhet me shpërndarjen prior:

$$
\theta \sim \text{Beta}(1,1)
$$

$\text{Beta}(1,1)$ është **shpërndarje uniforme** në $[0,1]$, që do të thotë se nuk favorizon asnjë vlerë të veçantë të $\theta$ para se të shohim të dhënat.

---

### 4.3. Posteriori – pas vlerësimeve

Duke përdorur rregullin e Bayes:

$$
P(\theta \mid \text{të dhënat}) \propto P(\text{të dhënat} \mid \theta)\, P(\theta)
$$

Për prior $\text{Beta}(1,1)$ dhe likelihood Bernoulli, rezultati standard është:

$$
\theta \mid \text{të dhënat} \sim \text{Beta}(1 + s,\ 1 + f)
$$

ku:

- $s$ = numri i review-eve **pozitive**
- $f = n - s$ = numri i review-eve **negative**

---

### 4.4. Rregulli i Laplace-it

Pyetja kryesore:

> **Cili është probabiliteti që review-i i ardhshëm të jetë pozitiv?**

Në qasjen bayesiane, ky probabilitet është **vlera e pritur (expected value)** e $\theta$-s sipas shpërndarjes posterior.

Për një shpërndarje $\text{Beta}(\alpha, \beta)$ vlen:

$$
\mathbb{E}[\theta] = \frac{\alpha}{\alpha + \beta}
$$

Në rastin tonë:

- $\alpha = 1 + s$
- $\beta = 1 + f$
- $n = s + f$

Prandaj:

$$
\mathbb{E}[\theta \mid \text{të dhënat}]
= \frac{1 + s}{1 + s + 1 + f}
= \frac{s + 1}{n + 2}
$$

Ky është **Rregulli i Laplace-it**:

$$
\boxed{p_{\text{Laplace}} = \frac{s + 1}{n + 2}}
$$

Interpretim:

- sikur të shtonim **1 sukses imagjinar** dhe **1 dështim imagjinar**,  
- kjo shmang vlerësimet ekstreme (0% apo 100%) kur kemi pak të dhëna.

---

## 5. Laplasi për tre ofruesit

### 5.1. Ofruesi 1

Të dhënat për Ofruesin 1:

- $s_1 = 10$ review pozitive  
- $n_1 = 10$ review gjithsej  

Formula:

$$
p_1 = \frac{s_1 + 1}{n_1 + 2}
     = \frac{10 + 1}{10 + 2}
     = \frac{11}{12}
     \approx 0.9167
$$

Pra probabiliteti që review-i i ardhshëm të jetë pozitiv është ≈ **91.67%**.

---

### 5.2. Ofruesi 2

Ofruesi 2:

- 50 vlerësime  
- 96% pozitive  

Numri i review-eve pozitive:

$$
s_2 = 0.96 \cdot 50 = 48
$$

Të dhënat:

- $s_2 = 48$  
- $n_2 = 50$  

Formula:

$$
p_2 = \frac{s_2 + 1}{n_2 + 2}
    = \frac{48 + 1}{50 + 2}
    = \frac{49}{52}
    \approx 0.9423
$$

Pra probabiliteti që review-i i ardhshëm të jetë pozitiv është ≈ **94.23%**.

---

### 5.3. Ofruesi 3

Ofruesi 3:

- 200 vlerësime  
- 93% pozitive  

Numri i review-eve pozitive:

$$
s_3 = 0.93 \cdot 200 = 186
$$

Të dhënat:

- $s_3 = 186$  
- $n_3 = 200$  

Formula:

$$
p_3 = \frac{s_3 + 1}{n_3 + 2}
    = \frac{186 + 1}{200 + 2}
    = \frac{187}{202}
    \approx 0.9257
$$

Pra probabiliteti që review-i i ardhshëm të jetë pozitiv është ≈ **92.57%**.


---

### 5.4. Krahasimi i rezultateve

| Ofruesi | $s$ (pozitive) | $n$ (totali) | Naive $s/n$ | Laplace $(s+1)/(n+2)$ | Laplace (%) |
|--------:|----------------|--------------|-------------|------------------------|-------------|
| 1       | 10             | 10           | 1.0000      | 0.9167                 | 91.67%      |
| 2       | 48             | 50           | 0.9600      | 0.9423                 | 94.23%      |
| 3       | 186            | 200          | 0.9300      | 0.9257                 | 92.57%      |

Renditja sipas probabilitetit të Laplace-it:

$$
p_2 > p_3 > p_1
$$

Prandaj:

> **Ofruesi 2** është zgjedhja më e arsyeshme sipas *Laplace’s rule of succession*.

---

## 6. Kodi i plotë në Python

Më poshtë është kodi i plotë i Python që përdor Rregullin e Laplace-it për tre produktet.  
Kodi është pa komente brenda; komentet dhe shpjegimet janë më poshtë, të ndara në **blloqe**.

```python
def probabiliteti_laplace(s_pozitive, n_total):
    return (s_pozitive + 1) / (n_total + 2)


produktet = {
    "Produkt 1": {"pozitive": 10, "total": 10},
    "Produkt 2": {"pozitive": 48, "total": 50},
    "Produkt 3": {"pozitive": 186, "total": 200},
}

print("\n--- Rezultatet duke përdorur Rregullin e Laplace-it ---\n")

for emri, data in produktet.items():
    laplace = probabiliteti_laplace(data["pozitive"], data["total"])
    print(f"{emri}: probabiliteti ≈ {laplace:.4f}    ({laplace * 100:.2f}%)")

me_i_miri = max(
    produktet,
    key=lambda p: probabiliteti_laplace(
        produktet[p]["pozitive"], produktet[p]["total"]
    ),
)

print(f"\nSipas Laplace-it, produkti më i besueshëm është: {me_i_miri}\n")


### 2) Struktura e të dhënave për produktet

```python
produktet = {
    "Produkt 1": {"pozitive": 10, "total": 10},
    "Produkt 2": {"pozitive": 48, "total": 50},
    "Produkt 3": {"pozitive": 186, "total": 200},
}
```

- **Çfarë bën:** Ruaj për secilin produkt numrin e vlerësimeve pozitive (`pozitive`) dhe totalin e vlerësimeve (`total`).
- **Pse duhet:** Kjo strukturë e thjeshtë (dictionary me nën-dictionary) e bën të lehtë iterimin dhe aplikimin e të njëjtit funksion për të gjitha produktet.

### 3) Mesazhi hyrës

```python
print("\n--- Rezultatet duke përdorur Rregullin e Laplace-it ---\n")
```

- **Çfarë bën:** Shton një titull të thjeshtë në terminal përpara rezultateve.
- **Pse duhet:** Rrit lexueshmërinë e output-it, sidomos kur krahasohen shumë produkte.

### 4) Llogaritja dhe shfaqja e probabiliteteve

```python
for emri, data in produktet.items():
    laplace = probabiliteti_laplace(data["pozitive"], data["total"])
    print(f"{emri}: probabiliteti ≈ {laplace:.4f}    ({laplace * 100:.2f}%)")
```

- **Çfarë bën:** Itëron mbi të gjitha produktet, llogarit probabilitetin e Laplace-it dhe e shfaq si numër dhe përqindje.
- **Pse duhet:** Kjo bllok i jep secilit produkt një vlerësim të drejtë që reflekton si përqindjen ashtu edhe numrin e vlerësimeve.

### 5) Gjetja e produktit “më të mirë”

```python
me_i_miri = max(
    produktet,
    key=lambda p: probabiliteti_laplace(
        produktet[p]["pozitive"], produktet[p]["total"]
    ),
)
```

- **Çfarë bën:** Përdor `max` me një funksion `key` për të krahasuar produktet sipas probabilitetit të Laplace-it dhe kthen emrin e produktit me vlerën më të lartë.
- **Pse duhet:** Siguron renditjen automatike të produkteve bazuar në kriterin bayesian, pa shkruar logjikë shtesë.

### 6) Mesazhi përfundimtar

```python
print(f"\nSipas Laplace-it, produkti më i besueshëm është: {me_i_miri}\n")
```

- **Çfarë bën:** Shtyp emrin e produktit me probabilitetin më të lartë të review-it pozitiv.
- **Pse duhet:** Jep një konkluzion të qartë për përdoruesin.

