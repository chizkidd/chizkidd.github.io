---
layout: post
comments: true
title: "Premium Is King? A Simple Look at U.S. Airline Unit Economics"
excerpt: A rabbit hole into CASM-ex, TRASM, and the unit economics of eight U.S. airlines. Why having the lowest operating costs doesn't necessarily mean having the best economics.
date: 2026-11-04
mathjax: true
---

I was listening to an episode of [Cheeky Pint](https://www.youtube.com/watch?v=3Lc0xfW4r9Q), a podcast hosted by Stripe cofounder John Collison, featuring Scott Kirby, CEO of United Airlines. During the conversation, Kirby mentioned something called ***CASM-ex***. I'd never heard of it before, so naturally I went down a rabbit hole. The rabbit hole eventually led me to a question I found more interesting than the original acronym: **What actually makes an airline economically successful?**

> Is it having the lowest operating costs? Charging passengers more? Filling more seats? Or some combination of all three?

Let's start with the terminology.

## What is CASM-ex?

CASM stands for *Cost per Available Seat Mile*. It's a measure of how much it costs an airline to make one seat available for one mile of travel. An available seat mile (ASM) is simply one seat flown for one mile, whether that seat is occupied or empty. For example, imagine an airline operates a 100-seat aircraft on a 1,000-mile flight. That flight produces:

$$
100 \times 1{,}000 = 100{,}000 \text{ ASMs}
$$

If operating the flight costs $10,000, then:

$$
\text{CASM} = \frac{\$10{,}000}{100{,}000} = 10\text{ cents}
$$

> In other words, it costs the airline **10 cents** to make each seat-mile available.

**CASM-ex** takes this a step further. It's an airline industry metric that measures how much it costs an airline to fly one available seat for one mile, excluding fuel expenses. The "ex" means excluding certain expenses, typically fuel and sometimes other items such as special charges or profit sharing.

### Why exclude fuel?

Because market fuel prices fluctuate significantly and are largely outside an airline's direct control. Looking at costs without fuel can help isolate how efficiently an airline runs its underlying operation. But there's a catch: airlines don't all exclude exactly the same expenses. So CASM-ex is useful, but comparing it across airlines requires some care.

## What about TRASM?

TRASM stands for *Total Revenue per Available Seat Mile*. It measures how much revenue an airline generates for every seat-mile it makes available. The distinction is important: an airline doesn't just earn revenue from the ticket itself. It can also generate revenue from baggage fees, seat selection, loyalty programs, cargo, and other services. For example, if the same 100-seat aircraft flying 1,000 miles generates $15,000 in operating revenue:

$$
\text{TRASM} = \frac{\$15{,}000}{100{,}000} = 15\text{ cents}
$$

> So the airline generates **15 cents** in revenue for every seat-mile it makes available.

Because TRASM is measured per *available* seat mile, it bundles together fares, load factor, and ancillary revenue, so it can't tell us on its own which of those is doing the work. Now we have two useful metrics: 
* **CASM** tells us how much it costs to produce capacity, while 
* **TRASM** tells us how much revenue that capacity generates.

And putting them together gives us something interesting.

## The difference between revenue and cost

We can calculate the operating spread:

$$
\text{Operating Spread} = \text{TRASM} - \text{Total CASM}
$$

Using our example:

$$
15 - 10 = 5\text{ cents}
$$

> The airline generates **5 cents** more in operating revenue than it spends on operating expenses for every available seat-mile. 

In other words, the operating spread is operating income per available seat mile. This is not net profit. It doesn't account for interest, taxes, and other non-operating expenses. But it's a useful way to understand the economics of the airline's core operations. We can also express the spread as a share of revenue, which gives us the unit operating margin:

$$
\text{Unit Operating Margin} = \frac{\text{TRASM} - \text{Total CASM}}{\text{TRASM}} = \frac{5}{15} \approx 33\%
$$

This is useful because 5 cents of spread means something very different on an airline that earns 15 cents per seat-mile than on one that earns 10. So, how do actual airlines compare?

## Comparing eight U.S. airlines

I pulled the full-year 2025 figures for eight U.S. airlines: Delta, United, American, Southwest, Alaska, JetBlue, Frontier, and Allegiant. I wanted a mix of legacy network carriers, hybrid carriers, low-cost airlines, and ultra-low-cost airlines. All figures below are in cents per available seat mile, except unit operating margin, and are sorted by operating spread.

| Airline | Business model | TRASM (operating revenue / ASM) | CASM-ex | Fuel CASM | Total CASM | Operating Spread | Unit Operating Margin |
|:---|:---|---:|---:|---:|---:|---:|---:|
| Delta | Legacy network | 19.56 | 13.86 | 3.29 | 17.61 | **+1.95** | **10.0%** |
| United | Legacy network | 17.88 | 12.64 | 3.45 | 16.46 | **+1.42** | **7.9%** |
| Allegiant | Ultra-low-cost / leisure | 11.91 | 8.04 | 2.99 | 11.24 | **+0.67** | **5.6%** |
| American | Legacy network | 18.25 | 14.12 | 3.58 | 17.76 | **+0.49** | **2.7%** |
| Alaska | Network / hybrid | 15.32 | 11.42 | 3.10 | 14.99 | **+0.33** | **2.2%** |
| Southwest | Low-cost | 15.59 | 12.32 | 2.91 | 15.35 | **+0.24** | **1.5%** |
| Frontier | Ultra-low-cost | 9.37 | 7.41 | 2.33 | 9.74 | **-0.37** | **-3.9%** |
| JetBlue | Hybrid | 13.94 | 11.20 | 3.16 | 14.51 | **-0.57** | **-4.1%** |

> _**Sources:** FY2025 company financial reports and earnings releases. Figures are rounded. Unit operating margin is calculated as operating spread divided by TRASM, using the rounded figures in the table. Delta's revenue and cost figures are adjusted to remove third-party refinery activity. Allegiant's operating revenue per ASM is calculated using airline-only operating revenue divided by total system ASMs. Alaska's figures reflect Alaska Air Group, including Hawaiian Airlines. CASM-ex definitions differ by airline._

---

<br>

One important clarification: **CASM-ex plus fuel CASM does not necessarily equal total CASM.** CASM-ex excludes different additional expenses depending on the airline. For example, Alaska also excludes freighter costs and special items, while JetBlue excludes other non-airline expenses and special items. Allegiant excludes special charges. The total CASM column, by contrast, represents total operating cost per seat-mile. However, it isn't perfectly comparable either, since some airlines make adjustments to their reported figures. Delta's, for example, is adjusted to remove third-party refinery activity. A few things jump out.

### 1. The cheapest airline doesn't necessarily win

Look at Frontier and Allegiant, two ultra-low-cost carriers. Both have relatively low CASM-ex figures. Frontier's CASM-ex is just 7.41 cents, while Allegiant's is 8.04 cents. Compare that with Delta at 13.86 cents and United at 12.64 cents. At first glance, Frontier and Allegiant appear to have a significant cost advantage. But look at the revenue side. Frontier generates only 9.37 cents in revenue per available seat mile against total operating costs of 9.74 cents per available seat mile. That's a negative operating spread, or a unit operating margin of about -3.9%. Allegiant, on the other hand, generates 11.91 cents against costs of 11.24 cents. So despite having higher unit costs than Frontier, Allegiant has a positive operating spread, and a unit operating margin of about 5.6%. This is the interesting part: **having the lowest cost base doesn't automatically translate into the strongest economics.** An airline also needs to generate enough revenue to cover those costs.

### 2. Premium is king?

Now look at the top of the table. Delta and United have the two strongest operating spreads, at 1.95 and 1.42 cents respectively. Both are legacy network carriers with significant premium offerings, loyalty programs, corporate travel exposure, and extensive route networks. This raises an interesting possibility. Perhaps the advantage isn't simply about making air travel cheaper to produce. It's about making the product valuable enough that customers are willing to pay more for it. Premium cabins, better schedules, loyalty benefits, airport lounges, and differentiated customer experiences can all contribute to stronger revenue generation. Of course, these airlines also have higher costs. Premium cabins take up space, lounges require investment, and a large network comes with operational complexity. But the numbers suggest that generating more revenue per seat-mile can compensate for a higher cost base. Delta is a particularly interesting example. Its CASM-ex is substantially higher than Frontier's, yet its revenue generation is also much stronger. It isn't a perfectly clean story, though. American is also a legacy carrier with a premium product, yet its operating spread is just 0.49 cents, below Allegiant's 0.67. Measured as a margin, the gap is clearer: about 2.7% for American versus about 5.6% for Allegiant.

### 3. Revenue matters just as much as cost discipline

Southwest and Alaska offer another interesting comparison. Their operating spreads are positive, but considerably narrower than Delta's and United's. Southwest's CASM-ex of 12.32 cents is close to United's 12.64, but it earns about 2.3 cents less revenue per available seat mile. JetBlue is also worth looking at. Its 2025 operating revenue per ASM of 13.94 cents fell short of its total operating cost of 14.51 cents. That produced a negative operating spread. This doesn't mean JetBlue has a fundamentally broken business model. These are results from one financial year, and airlines can have significant differences in network structure, fleet composition, route length, and other operating conditions. But it does illustrate how difficult the economics can become when revenue generation doesn't keep pace with operating costs.

## A few caveats

There are a few things worth keeping in mind before drawing too many conclusions from this comparison.

* **First, CASM-ex isn't standardized.** Different airlines exclude different expenses. Some exclude only fuel and special charges, while others also exclude profit sharing, freighter costs, or non-airline expenses. *That's why I wouldn't use CASM-ex alone to rank airlines by efficiency.*

* **Second, airline business models matter.** A network carrier operating long-haul international routes has a very different cost and revenue structure from an ultra-low-cost carrier flying leisure routes. Stage length, aircraft type, passenger mix, load factor, and network composition all affect unit economics.

* **Third, this is a snapshot, not a verdict.** These figures represent full-year 2025 performance. They don't establish that one business model will always outperform another, nor do they prove that premium offerings alone explain the differences. And finally, operating spread isn't net profit. It's a useful starting point for understanding the economics of the core airline operation, not the whole financial picture.

## Final thoughts

I started this rabbit hole trying to understand an acronym I'd heard on a podcast, and ended up with a much more interesting question about business strategy. It's tempting to think that the best airline is the one that can operate at the lowest cost. But looking at these numbers, that's clearly not the whole story. **Frontier has the lowest reported CASM-ex in this comparison, but it also has a negative operating spread. Delta has a much higher cost base, yet generates the strongest spread.** The ability to control costs matters. But so does the ability to generate revenue. Perhaps the real competitive advantage isn't simply figuring out how to fly a seat for less. It's figuring out how to make that seat worth more.

> And that brings me back to the original question: **Maybe premium really is king.**

---

## References

1. [Delta Air Lines 2025 Annual Report](https://s2.q4cdn.com/181345880/files/doc_downloads/annual/2026/DAL-12-31-2025-10K.pdf)
2. [United Airlines 2025 Full-Year Financial Results](https://ir.united.com/static-files/a37e057c-8ccf-41a8-ac93-4263f0527cc1)
3. [American Airlines 2025 Form 10-K](https://www.sec.gov/Archives/edgar/data/4515/000000620126000014/aal-20251231.htm)
4. [Southwest Airlines 2025 Form 10-K](https://investors.southwest.com/sec-filings/all-sec-filings/content/0000092380-26-000004/luv-20251231.htm)
5. [Alaska Air Group 2025 Form 10-K](https://www.sec.gov/Archives/edgar/data/766421/000076642126000010/alk-20251231.htm)
6. [JetBlue 2025 Full-Year Financial Results](https://ir.jetblue.com/news/news-details/2026/JetBlue-Announces-Fourth-Quarter-2025-Results/default.aspx)
7. [Frontier Airlines 2025 Form 10-K](https://s21.q4cdn.com/257736825/files/doc_financials/2025/q4/0001670076-26-000020.pdf)
8. [Allegiant 2025 Full-Year Financial Results](https://newsroom.allegiantair.com/press-releases/press-release-details/2026/ALLEGIANT-TRAVEL-COMPANY-FOURTH-QUARTER-AND-FULL-YEAR-2025-FINANCIAL-RESULTS/)