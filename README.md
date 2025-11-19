# Laplace’s Rule of Succession – Shembulli i rating-eve ne Amazon (Projekt ne Python)

---

## 1. Qellimi i projektit

Qellimi i ketij projekti eshte qe, duke perdorur **Rregullin e Laplace-it (Laplace’s rule of succession)** ne kuader te probabilitetit bayesian, te vendosim:

> Nga cili ofrues ne Amazon eshte me e arsyeshme te blihet produkti, kur cmimi eshte pothuajse i njejte, por ndryshojne **rating-et** (pozitiv / negativ) dhe **numri i vleresimeve**?

Hapat kryesore:

1. Pershkrimi i sakte i problemit.
2. Ndertimi i nje modeli probabilistik (Bernoulli + Beta).
3. Derivimi i formulas se Laplace-it.
4. Zbatimi i formulas ne **Python**.
5. Analiza e detajuar e kodit, e ndare ne **blloqe**.

Ky README eshte i dizajnuar qe te mund te lexohet edhe nga dikush qe nuk ka shume njohuri paraprake ne probabilitet ose Python.

---

## 2. Pershkrimi i problemit (Detyra)

Amazon sugjeron **tre ofrues** te nje produkti:

- Cmimet: praktikisht te njejta.
- Vendimi duhet te merret vetem nga **vleresimet e klienteve**.

Te dhenat:

- **Ofruesi 1** – 10 vleresime, 100% pozitive  
- **Ofruesi 2** – 50 vleresime, 96% pozitive  
- **Ofruesi 3** – 200 vleresime, 93% pozitive  

Detyra:

> Nga cili ofrues duhet blere produkti, nese synohet ai me **probabilitetin me te madh** qe review-i i ardhshem te jete pozitiv?

Ketu nuk mjafton vetem perqindja naive. Duhet te merret parasysh edhe **numri i vleresimeve**. Kjo behet ne menyre elegante me **Laplace’s Rule of Succession**.

---

## 3. Qasja naive: frekuenca \(s/n\)

Qasja me e thjeshte per te vleresuar probabilitetin e review-it pozitiv eshte:

\[
\hat{p} = \frac{s}{n}
\]

ku:

- \(s\) = numri i review-eve **pozitive**,  
- \(n\) = numri i review-eve **totale**.

Per ofruesit:

1. Ofruesi 1:  
   \[
   \hat{p}_1 = \frac{10}{10} = 1.00 = 100\%
   \]
2. Ofruesi 2:  
   \[
   \hat{p}_2 = \frac{48}{50} = 0.96 = 96\%
   \]
3. Ofruesi 3:  
   \[
   \hat{p}_3 = \frac{186}{200} = 0.93 = 93\%
   \]

Sipas kesaj logjike naive, do te zgjidhnim **Ofruesin 1** (100%).

Problemi:

- Ofruesi 1 ka vetem **10 vleresime** → shume pak te dhena.
- Mjafton vetem nje review negativ dhe shkalla bie ne 90%.
- Ofruesi 3 ka **200 vleresime** → shume me shume informata, megjithese perqindja eshte me e ulet.

Pra, qasja naive nuk e dallon qarte ndryshimin ndermjet:

- **perqindjes se suksesit** (s/n),
- dhe **volumit te te dhenave** (n).

Duhet nje qasje me e kujdesshme – ketu futet **Bayes** dhe **Laplace**.

---

## 4. Modeli probabilistik bayesian

### 4.1. Modelimi i review-eve (Bernoulli)

Supozimet:

- review pozitiv → e modelojme si 1,  
- review negativ → e modelojme si 0.

Secili review eshte eksperiment **Bernoulli** me probabilitet:

\[
P(\text{review pozitiv}) = \theta, \quad
P(\text{review negativ}) = 1 - \theta
\]

Per secilin ofrues kemi nje \(\theta\) te ndryshem:

- \(\theta_1\) – probabiliteti i review pozitiv per Ofruesin 1,  
- \(\theta_2\) – probabiliteti i review pozitiv per Ofruesin 2,  
- \(\theta_3\) – probabiliteti i review pozitiv per Ofruesin 3.

Nese per nje ofrues kemi:

- \(s\) review pozitive,
- \(f\) review negative,
- \(n = s + f\) review gjithsej,

atehere **likelihood** i te dhenave (probabiliteti i te dhenave nese \(\theta\) eshte e dhene) eshte:

\[
P(\text{te dhenat} \mid \theta) = \theta^{s}(1 - \theta)^{f}
\]

---

### 4.2. Prior – cfare dim para te dhenave?

Para se te vleresohen review-et, supozojme qe:

- nuk kemi arsye te dyshojme qe produkti eshte shume i mire apo shume i keq,
- te gjitha vlerat e \(\theta\) ne \([0, 1]\) jane **po aq te mundshme**.

Kjo pershkruhet me shperndarjen:

\[
\theta \sim \text{Beta}(1, 1)
\]

Beta(1,1) eshte **shperndarje uniforme** ne \([0,1]\).

---

### 4.3. Posteriori – pas vleresimeve

Duke perdorur rregullin e Bayes:

\[
P(\theta \mid \text{te dhenat}) \propto P(\text{te dhenat} \mid \theta)\, P(\theta)
\]

Per prior Beta(1,1) dhe likelihood Bernoulli, rezultati standard eshte:

\[
\theta \mid \text{te dhenat} \sim \text{Beta}(1 + s,\ 1 + f)
\]

ku:

- \(s\) = numri i review-eve pozitive,  
- \(f = n - s\) = numri i review-eve negative.

---

### 4.4. Rregulli i Laplace-it

Pyetja kryesore:

> Cili eshte probabiliteti qe **review-i i ardhshem** te jete pozitiv?

Ne qasjen bayesiane, ky probabilitet eshte **vlera e pritur** (expected value) e \(\theta\)-s sipas shperndarjes posterior.

Per nje shperndarje Beta(\(\alpha,\beta\)) vlen:

\[
\mathbb{E}[\theta] = \frac{\alpha}{\alpha + \beta}
\]

Ne rastin tone:

- \(\alpha = 1 + s\),
- \(\beta = 1 + f\),
- dhe \(n = s + f\).

Prandaj:

\[
\mathbb{E}[\theta \mid \text{te dhenat}] =
\frac{1 + s}{1 + s + 1 + f} =
\frac{s + 1}{n + 2}
\]

Ky eshte **Rregulli i Laplace-it**:

\[
\boxed{p_{\text{Laplace}} = \frac{s + 1}{n + 2}}
\]

Interpretim:

- sikur te shtonim **1 sukses imagjinar** dhe **1 deshtim imagjinar**;
- kjo ben qe te mos kemi kurrre probabilitet 0% apo 100% me numer te vogel te dhenash.

---

## 5. Laplace per tre ofruesit

### 5.1. Ofruesi 1

Te dhenat:

- \(s_1 = 10\) pozitive,  
- \(n_1 = 10\) gjithsej.

Formula:

\[
p_1 = \frac{s_1 + 1}{n_1 + 2}
    = \frac{10 + 1}{10 + 2}
    = \frac{11}{12}
    \approx 0.9167
\]

Pra probabiliteti i review-it te ardhshem pozitiv eshte ≈ **91.67%**.

---

### 5.2. Ofruesi 2

Ofruesi 2:

- 50 vleresime,
- 96% pozitive.

Numri i review-eve pozitive:

\[
s_2 = 0.96 \cdot 50 = 48
\]

Te dhenat:

- \(s_2 = 48\),
- \(n_2 = 50\).

Formula:

\[
p_2 = \frac{s_2 + 1}{n_2 + 2}
    = \frac{48 + 1}{50 + 2}
    = \frac{49}{52}
    \approx 0.9423
\]

Pra probabiliteti i review-it te ardhshem pozitiv eshte ≈ **94.23%**.

---

### 5.3. Ofruesi 3

Ofruesi 3:

- 200 vleresime,
- 93% pozitive.

Numri i review-eve pozitive:

\[
s_3 = 0.93 \cdot 200 = 186
\]

Te dhenat:

- \(s_3 = 186\),
- \(n_3 = 200\).

Formula:

\[
p_3 = \frac{s_3 + 1}{n_3 + 2}
    = \frac{186 + 1}{200 + 2}
    = \frac{187}{202}
    \approx 0.9257
\]

Pra probabiliteti i review-it te ardhshem pozitiv eshte ≈ **92.57%**.

---

### 5.4. Krahasimi i rezultateve

| Ofruesi | \(s\) (pozitive) | \(n\) (totali) | Naive \(s/n\) | Laplace \((s+1)/(n+2)\) | Laplace (%) |
|--------:|------------------|----------------|---------------|-------------------------|-------------|
| 1       | 10               | 10             | 1.0000        | 0.9167                  | 91.67%      |
| 2       | 48               | 50             | 0.9600        | 0.9423                  | 94.23%      |
| 3       | 186              | 200            | 0.9300        | 0.9257                  | 92.57%      |

Renditja sipas probabilitetit te Laplace-it:

\[
p_2 > p_3 > p_1
\]

Prandaj:

> **Ofruesi 2** eshte zgjedhja me e arsyeshme sipas Laplace’s rule of succession.

---

## 6. Kodi i plote ne Python

Me poshte eshte kodi i plote i Python qe perdor Rregullin e Laplace-it per tre produktet.  
Kodi eshte pa komente brenda; komentet dhe shpjegimet jane me poshte, te ndara ne **blloqe**.

```python
def probabiliteti_laplace(s_pozitive, n_total):
    return (s_pozitive + 1) / (n_total + 2)


produktet = {
    "Produkt 1": {"pozitive": 10, "total": 10},
    "Produkt 2": {"pozitive": 48, "total": 50},
    "Produkt 3": {"pozitive": 186, "total": 200},
}

print("\n--- Rezultatet duke perdorur Rregullin e Laplace-it ---\n")

for emri, data in produktet.items():
    laplace = probabiliteti_laplace(data["pozitive"], data["total"])
    print(f"{emri}: probabiliteti ≈ {laplace:.4f}    ({laplace * 100:.2f}%)")

me_i_miri = max(
    produktet,
    key=lambda p: probabiliteti_laplace(
        produktet[p]["pozitive"], produktet[p]["total"]
    ),
)

print(f"\nSipas Laplasit, produkti me i besueshem eshte: {me_i_miri}\n")

## Shpjegimi i kodit në blloqe

### 1) Funksioni `probabiliteti_laplace`

```python
def probabiliteti_laplace(s_pozitive, n_total):
    return (s_pozitive + 1) / (n_total + 2)
```

- **Çfarë bën:** Llogarit probabilitetin e review-it të ardhshëm që të jetë pozitiv, duke aplikuar formulën e Laplace-it. 
- **Pse duhet:** Ky funksion përfshin “suksesin” dhe “dështimin” imagjinar (+1 në numërues dhe +2 në emërues) për të amortizuar efektin e mostrës së vogël.

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
print(f"\nSipas Laplasit, produkti më i besueshëm është: {me_i_miri}\n")
```

- **Çfarë bën:** Shtyp emrin e produktit me probabilitetin më të lartë të review-it pozitiv.
- **Pse duhet:** Jep një konkluzion të qartë për përdoruesin.

