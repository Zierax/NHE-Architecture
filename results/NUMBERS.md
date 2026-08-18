# الأرقام المرجعية الموحّدة (Canonical Numbers) — NHE-Architecture

آخر تحديث: 2026-08-18. هذا الملف هو المصدر الوحيد المعتمد للأرقام؛ أي وثيقة أو عرض يستشهد بأرقام مخالفة يجب أن يتطابق مع هذا الجدول **مع تسمية البروتوكول** (greedy مقابل sampled).

## بروتوكولا القياس
| البروتوكول | الوصف | الملف المرجعي |
|---|---|---|
| **greedy** | فك حتمي (argmax)، عنصر واحد لكل سؤال | `results/eval_{topic}_{mask}.json` (بدون `_s5`) |
| **sampled (majority)** | temp=0.9, top_p=0.9، 5 بذور (1000+i*100+s)، تصويت الأغلبية لكل عنصر | `results/eval_{topic}_{mask}_s5.json` |

> ملاحظة توحيد: أي قيمتين لمقارنة واحدة يجب أن تكونا من نفس البروتوكول. مثال التعارض المحلول: `12.96%→5.56%` هي **greedy** لـk128_wrong، و`0.093` هي **sampled majority** لنفس القناع — كلاهما صحيح لكنهما بروتوكولان مختلفان.

## الجدول المعتمد — أفريقيا (54 سؤالًا)

| التدخل | greedy hall | sampled-majority hall | أضرار جانبية (greedy) | الدليل |
|---|---|---|---|---|
| baseline | 0.130 (7/54) | 0.167 (9/54) | — | `eval_africa_baseline.json`, `eval_africa_baseline_s5.json` |
| إحصائي d_mean/d_var | 0.130 (NULL) | — | لا شيء | `eval_africa_k32_mean.json` |
| **k32_midwrong** (AtP، wrong-only، طبقات 8–17) | 0.093 (5/54) — إصلاح 3 (Eswatini/Gambia/Senegal) **+ كسر South Sudan (Juba→Bor)** | **0.111 (6/54)، p=0.017 (McNemar)، صافي +15 [5,24]** | أوروبا: 0.000→0.000 (لا ضرر) | `eval_africa_k32_midwrong.json`, `_s5`, `eval_europe_k32_midwrong.json` |
| **k128_wrong** | **0.056 (3/54)** — أقوى خفض greedy | **0.093 (5/54)، p=0.003، صافي +20 [8,33]** | **أوروبا: 0.000→0.091 (4/44 مكسورة)، p<0.001** | `eval_africa_k128_wrong.json`, `eval_europe_k128_wrong.json`, `_s5` |
| **زمني مبكر منوَّع** (jump_max_early_L19، t90، نافذة ≤5، قناع k32_midwrong) | **0.074 (4/54)** — إصلاح 3، **صفر كسور**، إطلاقات 7/54 | بذرة واحدة s1000: 0.130→0.111 (إصلاح 2: Eswatini/Gambia + **كسر Burundi: Gitega→Bujumbura**) | أوروبا: 0 إطلاقات، 0.000؛ آسيا/أمريكا/عناصر = baseline تمامًا | `eval_runtime_africa_jump_gt_L19_t90_mask.json`, `_s1000`, `eval_runtime_europe_*.json` |

## كاشفات الكشف (أفريقيا، greedy)
| الكاشف | AUC (LOSO) | ملاحظة |
|---|---|---|
| jump_max_L18 (كامل التوليد) | 0.860 | يطلق بعد الالتزام (خامل تدخليًا) |
| jump_max_early_L19 (أول 10 توكينات) | 0.742 | يطلق قبل الالتزام على 3/7 — الفعال تدخليًا |
| probe L10 (logistic) | 0.672 | أضعف؛ لا ينتقل بين بروتوكولات الفك |

## السقف المؤكد
4/7 هلوسات greedy (Cape Verde، Equatorial Guinea، Gabon، Guinea) تلتزم "بهدوء" — لا عتبة jitter تلتقطها قبل الالتزام ولا قناع k32_midwrong يصلحها. لا عتبة ولا قناع من هذه العائلة تتجاوز هذا السقف.

## قيود صادقة
- تصنيف المطابقة النصية متساهل (مثال: "Salaffaire...and Praia" تُعد صحيحة؛ Senegal عُدّت إصلاحًا وهو hedge "Diou... While Dakar is the largest city") — كل انقلاب في النتائج النهائية فُحص يدويًا.
- كل الأرقام لموديل واحد (Gemma 3 1B) على CPU؛ مقاسات النماذج الأخرى غير مغطاة.
- أُبلغ المستخدم: كل شيء على CPU، لا CUDA حتى الآن.