---
layout: post
comments: true
title: "Why AI Voice Assistants Fail Tonal Languages: ASR Bias in Igbo"
excerpt: A small evaluation of tone preservation in facebook/omniASR-CTC-1B on Igbo. On tonal clips, 3 out of 4 expected tone and dot-below marks are missing from the transcription, and other clips gain marks that were not in the reference. The pattern is consistent with the model leaning on text priors rather than acoustic tone, but it is one speaker and 21 clips.
date: 2026-03-04
mathjax: true
---

I grew up in an Igbo household in Northern Nigeria, that code-switched between English, Igbo, and Hausa almost unconsciously. Like many bilingual Nigerians, I've watched voice assistants and ASR systems get better and better at English while struggling with our languages. When Meta released omniASR claiming support for over 1,600 languages including Igbo, I was curious. Does "supported" mean it actually works?

Turns out, the answer is more complicated than I expected.

{% capture c %}
This post was corrected on 17 October 2026. Earlier versions mixed a 61.2% net loss with a 75.5% drop rate, said the confidence intervals did not overlap between categories (they do for most pairs), and described the model's added marks as proof of orthographic bias. The numbers are now recomputed with dropped and added marks counted per clip, the claims are softened to what one speaker and 21 clips can support, and word meanings for the akwa forms were removed pending a check against a standard Igbo reference.
{% endcapture %}
{% include callout.html type="note" title="Update (October 2026)" content=c %}

## The Problem: What Does "Language Support" Really Mean?

Here's the thing about Igbo: tone changes word meaning. The difference between "akwa", "akwà", "àkwà", and "ákwá" isn't just decorative accent marks. These are different words that happen to have the same consonants and vowels. The tone is the difference.

So when I saw that omniASR listed Igbo among its supported languages, I wanted to know: does it actually preserve these tonal distinctions? Or does "support" just mean "we trained on some Igbo data and hope for the best"?

## The Experiment: 21 Audio Samples

I designed a simple test. Using my iPhone Voice Memos app, I recorded 21 short audio clips in different categories:

**Tonal minimal pairs**: I said "akwa, akwa, akwa" three times with no tone, then "akwà, akwà, akwà" three times with low tone, then "àkwà, àkwà, àkwà" with low-low tone, and finally "ákwá, ákwá, ákwá" with high-high tone. Four distinct words, each repeated three times.

**Code-switching**: Phrases like "The ụlọ is beautiful" where I mix English and Igbo naturally, the way we actually speak.

**Place names and cultural terms**: Nigerian cities, Igbo food words, proverbs. The stuff that's probably not in training data.

**The key diagnostic test**: I spoke a sentence with deliberately flat intonation, no tonal variation at all. If the model is actually listening to tone in the audio, it shouldn't add tone marks to monotone speech.

Then I ran everything through omniASR and compared what I actually said to what it transcribed.

## The Results: 75% of Expected Marks Missing

The numbers were worse than I expected.

I counted every letter that carries a tone mark or a dot-below (ọ, ị, ụ) in my transcripts, and compared with what the model produced, clip by clip. On the six tonal clips, 37 of the 49 expected marks were missing from the model's output. That is 75.5%: three out of every four marks, gone. (The model also added 7 marks that weren't in my transcripts, so the net change is smaller, 61.2%, but the 37 missing ones are the story here.)

When I said the four different "akwa" words, the model output was: "akua akua akua akua akwa akwa akwa akua akwa ọkua ọkua ọkua". The output has no tone marks at all, and the four forms are not distinguished.

But here's what made me suspect the model isn't actually listening to tones: the monotone test. I spoke "O na-eri oji n'ututu" (He eats kolanut in the morning) with flat intonation, like a robot. The model transcribed it as "ọne rị ọjí nụ tútú", which has 3 tone marks and 4 dot-below letters that my transcript doesn't have.

If the model were using acoustic information to place diacritics, flat speech shouldn't attract tone marks. That is consistent with it doing something else: maybe using statistical patterns from training data to guess where diacritics go, rather than hearing them. I can't confirm that from one clip, and the dot-below letters may just be the standard spelling, since I wrote my transcript without dots.

{% capture c %}
**File 09:** Spoke "O na-eri oji n'ututu" with FLAT intonation  
**Expected:** 0 marks (no tonal variation in audio)  
**Result:** Output has 7 marks I didn't write: 3 tone marks, 4 dot-below letters  
<br>
This is **consistent with** orthographic bias, but it is one clip.
{% endcapture %}
{% include callout.html type="note" title="Key Diagnostic: The Monotone Test" content=c %}

### What the Data Shows

I created three visualizations to make the patterns clear.

![marks dropped vs added by category](/assets/images/2026/omniASR/fig1_drop_rate_by_category.png)

**Figure 1** shows, for each category, the share of expected marks that were dropped (dark) and the share that were added (grey). The tonal category jumps out at 75.5% dropped. But look at the cross-lingual category: only 11.1% dropped, while added marks equal 50% of the expected count (9 added against 18 expected). The model isn't only dropping marks, it's adding them where my transcripts have none. Showing both avoids the trap of netting them: the tonal category's net loss is 61.2% and the cross-lingual one is -38.9%, which hides how much is happening in each direction.

![char error rate vs drop rate](/assets/images/2026/omniASR/fig2_cer_vs_drop_rate.png)

**Figure 2** plots character error rate against the drop rate for each clip that had at least one expected mark. The tonal clips (red dots) sit at the high end of both: CER of about 48% to 53% and 50% to 100% of marks dropped. Across all 18 clips the correlation is weak (r = 0.36), but with so few clips and so few marks per clip I can't tell how much tone loss is separate from the model doing poorly overall.

![bootstrap confidence interval](/assets/images/2026/omniASR/fig3_bootstrap_95_ci.png)

**Figure 3** shows 95% confidence intervals for the drop rate in each category, from resampling clips. The tonal interval (57% to 90%) stays well above zero. The other categories only have five clips each, so their intervals are huge, and the tonal interval overlaps two of them. The clean statement is that the tonal clips drop a lot of marks, not that they drop more than every other category: only the contrast with the cross-lingual category clearly excludes zero.

## The Statistical Story

I'm not a statistician, but I know enough to be careful with small sample sizes. Twenty-one samples isn't huge. So I used bootstrap resampling (basically, randomly resampling my clips 10,000 times to get confidence intervals) to see how much the numbers could move.

For the tonal clips the drop rate was 75.5%, with a 95% interval of 57.1% to 89.7%. Even the lower end is a lot of missing marks. The caveat is that this tells you about noise in my dataset, not about Igbo speakers in general: six clips from one speaker, recorded in one session, are not a random sample.

I also created custom metrics because standard Character Error Rate treats tone marks the same as spacing errors. I track dropped marks, added marks, and combine them into a Diacritic Error Rate (DER). Turns out the model isn't just dropping marks. It's also adding marks that aren't in my transcripts, which is a whole different kind of problem.

## The Categories

Breaking down the errors helped me understand what's going wrong:

**Cross-lingual interference**: My transcripts for these clips (names, greetings, numbers, a proverb) have 18 marks, mostly dot-below letters. The model produced 25: 2 dropped, 9 added. Only 4 of its marks are tone marks, so part of this is spelling convention. It's consistent with the model applying orthographic patterns from other languages, but doesn't prove it.

**Code-switching boundary effects**: Net loss was only 14.3%, but that hides 7 of 14 expected marks dropped and 5 added. The English portions of code-switched sentences were mostly transcribed, with some errors, and the Igbo portions next to English lost marks. Something about language boundaries may be disrupting processing, though five clips can't say.

**Domain coverage**: Culturally specific terms (place names, food words) had the lowest net loss at 6.3%, but 5 of 16 expected marks were still dropped and 4 added, and character error rate was about 37%. The model gets some of the orthography right but many of the words wrong.

**Tonal collapse**: 37 of 49 expected marks dropped, 75.5%. This is the big one.

## Why This Matters

I keep coming back to the monotone hallucination test. If I were building a voice assistant for Igbo speakers and it's adding tones I didn't speak, that's not just an accuracy problem. It's an epistemological problem. The system is presenting confident outputs that have no acoustic basis.

Imagine you're dictating a text message in Igbo and the system confidently transcribes one word when you said another that differs only in tone. Not just a typo you can spot and fix. A different word that can make semantic nonsense but looks plausible.

{% capture c %}
75.5% of expected marks missing means:<br>
**3 out of 4** tone and dot marks disappear<br>
**Minimal pairs** collapse into the same few outputs<br>
**Tone-only forms** can't be told apart<br>
<br><br>
In English, this would be like dropping a large share of the consonants.
{% endcapture %}
{% include callout.html type="note" title="What 75% Missing Marks Means" content=c %}

This isn't just about transcription accuracy. It's about whether "supporting 1,600+ languages" means anything more than "we trained on data from 1,600+ languages and didn't check if it actually works for tonal distinctions."

## The Bigger Picture: Zeno's Paradox of Low-Resource Languages

There's a paper from EMNLP 2024 that talks about "The Zeno's Paradox of Low-Resource Languages." The basic idea: models keep claiming to support more and more languages, but the quality asymptote never actually reaches parity with high-resource languages. We get closer and closer, but never quite there.

Igbo is interesting because by speaker population (45 million people), it's not low-resource. But by model performance, it clearly behaves like one. The gap between coverage (we trained on Igbo data) and competence (the model preserves linguistically meaningful distinctions) is huge.

{% capture c %}
omniASR claims support for 1,600+ languages. Igbo has 45 million speakers, but on my tonal clips only 12 of 49 expected marks survived in count, and by an alignment estimate only 4 landed on the right letter with the right mark.<br><br>
**Coverage** (in training data) ≠ **Competence** (preserves meaning)
{% endcapture %}
{% include callout.html type="note" title="'Supported' ≠ Works Well" content=c %}


This makes me think about all the other languages in that 1,600+ list. How many of them have this same gap? How many communities are using systems that confidently produce nonsense because nobody with native speaker expertise has stress-tested them?

## What I Learned

**Small, targeted datasets can reveal problems big datasets hide.** I didn't need thousands of hours of audio. Twenty-one carefully designed samples were enough to show systematic failure modes.

**Native speaker expertise matters.** Automated metrics can't catch when one tonal form of a word is transcribed as another, because the character error rate can look fine. You need someone who speaks the language to know that the meaning changed. It also means my own transcripts are a single annotator's judgment, and a second annotator would make the results stronger.

**Bootstrap resampling shows how little a small sample pins down.** I was worried 21 samples was too few. The intervals confirmed that for most categories (they are huge) while showing the tonal result is robust within this dataset. Resampling clips, not individual marks, is what keeps them honest.

**The monotone test is a better diagnostic than I expected.** If marks are added to flat speech, that points toward text priors over acoustic conditioning. It is one clip, so it is a lead for a proper acoustic test, not a finding.

## The Technical Details

For anyone interested in replicating this:

- I used my iPhone for recording (Voice Memos app, M4A format)
- Ran inference through Google Colab with omniASR's official pipeline
- Computed bootstrap CIs with 10,000 resamples of clips (clip level, not mark level)
- Counted dropped and added marks per clip and combined them into a DER metric to separate tonal errors from general transcription errors
- All code, data, and analysis is on GitHub and HuggingFace

The whole analysis took about half a week of evening work. Most of that was iterating on the sample design and figuring out the right statistical approach. The actual recording and inference was maybe a day.

## What's Next

This is really just a proof of concept. To make stronger claims, I'd need:

- A second annotator for my transcripts
- Multi-speaker evaluation (10+ speakers across different Igbo dialects)
- Acoustic analysis (F0 contour tracking to verify what's actually in the audio)
- Comparative evaluation (does Whisper do better? What about Google's USM?)
- Fine-tuning experiments (can we fix this with targeted training data?)

I have ideas for all of these, but they're bigger projects. For now, I'm focused on documenting the blind spot and making the methodology replicable.

## Why I'm Sharing This

This started as curiosity about whether "multilingual" ASR systems actually work for the languages I grew up speaking. But it turned into something bigger.

There's a tendency in ML to treat "supporting" a language as a checkbox. Train on some data, add it to the model card, ship it. But languages aren't just data. They're how people communicate, how they think, how they preserve culture.

When voice assistants strip tone marks from Igbo, they're not just making transcription errors. They're normalizing a version of the language that doesn't preserve meaning. If every voice interface does this, what happens to how people write Igbo? Do they start thinking tone marks are optional because the AI doesn't use them?

I don't know the answers to these questions. But I think they're worth asking before we claim to "support" 1,600+ languages.

## Resources

If you want to explore the data or replicate the analysis:

- **Dataset:** [HuggingFace](https://huggingface.co/datasets/Chiz/omniASR-igbo-blindspots)
- **Code:** [GitHub](https://github.com/chizkidd/igbo-asr-tonal-evaluation)
- **Audio samples:** You can actually listen to the 21 clips and see the transcription failures yourself

The dataset is CC-BY-4.0 licensed, while the code is MIT licensed. If this is useful for your work, feel free to use it, cite it, and build on it.

## Final Thoughts

This project taught me something important: you don't need massive compute or huge datasets to find meaningful problems in ML systems. You just need to know where to look and what questions to ask.

As a native Igbo speaker, I knew what questions to ask. As someone learning ML, I knew how to design tests and interpret results. That combination turned out to be more valuable than I expected.

If you speak a language that's "supported" by these big multilingual models, I encourage you to test them. Record some minimal pairs. Try code-switching. See if the system actually works the way you use the language, not just the way it appears in training data.

You might be surprised what you find.


## Citation

If you found this work helpful, please consider citing it:
```bibtex
@article{obasi2026igboasr,
  title   = "When Your Voice Assistant Can't Hear Tones: Evaluating ASR Bias in Igbo",
  author  = "Obasi, Chizoba",
  journal = "chizkidd.github.io",
  year    = "2026",
  month   = "Mar",
  url     = "https://chizkidd.github.io/2026/03/04/igbo-asr-tonal-evaluation/"
}
```
