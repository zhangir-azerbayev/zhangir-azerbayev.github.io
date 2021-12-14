---
layout: post
title: Code Generation as Neurosymbolic Reasoning
---
Scaling pre-trained language models to ever larger datasets and parameter counts has yielded an astonishing ability to generate plausible-sounding text. Consider this GPT-3 completion of a sentence from the front page of today's *Financial Times.* 

> ***Prompt:* European gas futures surged 11 per cent on Monday after German foreign minister Annalena Baerbock said the Nord Stream 2 pipeline could not be permitted in its current form because it did not comply with EU law.**

 Russian President Vladimir Putin has been pushing for the pipeline since it was first proposed in 2015 by the Russian gas company Gazprom, which holds the majority stake in the project.

 The pipeline would add another 55 billion cubic metres of gas to Germany's existing import of 45 billion cubic metres of gas a year, which comes mainly from Russia. The pipeline would run in parallel to the existing Nord Stream 1 pipeline.

But the US and some EU member states oppose the project, arguing that it would increase Europe's dependence on Russian gas and would pose a threat to Ukraine's gas transit business. A number of EU members, including the Netherlands, Sweden, Finland and Denmark, have objected to the project on the grounds of violating the EU's Third Energy Package - a set of rules aimed at separating energy industries from the providers.

The pipeline is to stretch from the Russian port of Vyborg, near St Petersburg, to the coast of Germany. The German government is expected to make a decision on the project by the end of the year.
> 

Without knowing in advance, I would certainly not be able to tell this wasn't written by a human. Every sentence is grammatical, idiomatic, on-topic, and the passage demonstrates impressive general knowledge. If there are factual errors, none jump out at me. 

However, further investigation of GPT-3 shows the picture is not always so bright. One particular area where transformer-based language models have persistently struggled is logical, mathematical, and algorithmic reasoning. Despite being able to write convincing newspaper articles, GPT-3 cannot do two-digit multiplication. 

> ***Prompt:* 12 x 37 =** 
4,944
> 

Here is another questions humans find trivial where GPT-3 fails: 

> ***Prompt:* 
Q:John has 4 apples and 7 oranges in a basket, on the way home he lost 2 apples. How many pieces of fruit does John have?**
A: John has 5 apples and 5 oranges in a basket.
> 

GPT-3 commits two distinct errors here. First, it fails to properly calculate the number of apples and oranges, and then fails to unify the information about apples and oranges into a single count of fruits. We might speculate we are pushing GPT-3 too hard when ask it to do both commonsense inference and math at the same time. But alas, GPT-3 does not fare much better at *a priori* reasoning. 

> **Prompt: A foo and a bar are a kind of baz. John has 4 foo and 7 bar in a basket, on the way home he lost 2 foo. How many baz does he have?**
John has 3 baz in a basket.
> 

At this point, many authors throw up their hands, conclude that next-token prediction will never produce anything more than statistical slurry, and become interested in things like [neural module networks](https://arxiv.org/abs/1511.02799) or [computational cognitive science](https://arxiv.org/abs/1604.00289). However, I hope to convince you that, at least in this case, [the bitter lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) prevails: we do not need any major algorithmic innovations for transformers to make tremendous progress on natural language reasoning. 

OpenAI Codex, a model with the same architecture as GPT-3 but trained on public GitHub repos, was pitched by OpenAI as a programming assistant that automates tedious tasks. However, the implications of language models trained on code are far more profound than this single narrow application. I propose that language models trained on code can leverage programming languages as *universal logical forms* for doing natural language reasoning. 

### Code generation as neurosymbolic computing

For Codex, arbitrary multiplication problems are trivial because they are simply offloaded to the language. In all the following examples, the docstring is the prompt and the following program is the Codex completion. 

```python
"""
Calculate the product of 1234679 and 219837321
"""

def multiply(a, b):
    return a * b

print(multiply(1234679, 219837321))
```

This is of course also the case not just the case for arithmetic operations, but many algorithmic tasks such sorting and searching. This example is suggestive of interpreting the text-to-code models not just as programming assistants, but a powerful method for neurosymbolic computation. The transformer decomposes a task into discrete logical steps, and the programming language executes rules-based computations. 

### Scaffolding for multi-step reasoning

Let's revisit the following prompt

> **John has 4 apples and 7 oranges in a basket, on the way home he lost 2 apples. How many pieces of fruit does John have?**
> 

In order for a transformer to solve this prompt, it must at least take these four inferences: 

1. An apple is a kind of fruit. 
2. An orange is a kind of fruit. 
3. "Lost" corresponds to subtraction 
4. 4 + 7 - 2 = 9

We saw that GPT-3 is unable to make any of these inferences, so if code generation merely offered the ability to black-box arithmetic, Codex would still fail at (1)-(3). However, we find that Codex somehow does have commonsense reasoning capabilities beyond GPT-3. 

```python
"""
John has 4 apples and 7 oranges in a basket. 
On the way home he loses 2 oranges. How many fruit does he have? 
"""
def num_fruit(apples, oranges): 
	return apples + oranges - 2

print(num_fruit(4, 7))
```

Here are some more examples of Codex succeeding on natural language reasoning tasks GPT-3 fails at. 

```python
"""
A foo and a bar are a kind of baz. John has 4 foo and 7 bar in a basket, 
on the way home he lost 2 foo. How many baz does he have?
"""
def baz_count(foo, bar, lost_foo): 
	return foo + bar - lost_foo 

print(baz_count(4, 7, 2))
```

```python
"""
China won 4 gold medals, 3 silver medals, and 8 bronze medals. The USA won 5 gold medals, 1 silver medal, and 10 bronze medals. Who won more medals?
"""

# China
china_gold = 4
china_silver = 3
china_bronze = 8

# USA
usa_gold = 5
usa_silver = 1
usa_bronze = 10

# Total medals
china_total = china_gold + china_silver + china_bronze
usa_total = usa_gold + usa_silver + usa_bronze

print("China won", china_total, "medals")
print("USA won", usa_total, "medals")

if china_total > usa_total:
    print("China won more medals")
elif china_total < usa_total:
    print("USA won more medals")
```

Admittedly, Codex still has pronounced failure modes. For one, Codex starts making mistakes at large recursion depths. 

```python
"""
John has three apples. Mary gave John two more apples. John ate one apple. Then John lost one more apple. Then Mary gave John 5 apples. Right now, John has …
"""

john_apples = 3
mary_apples = 2
john_apples = john_apples + mary_apples
john_apples = john_apples - 1
mary_apples = mary_apples + 5
john_apples = john_apples + mary_apples
print(john_apples) # prints 11, correct answer is 8 
```

Occasionally Codex's errors are just bizarre. In the next example, for whatever reason, Codex decides to generate a long comment unrelated to the prompt. 

```python
"""
John spent 1 hour running, 2 hours playing soccer, and 1 hour reading. Calculate the amount of hours John spent exercising. 
"""

# Exercise
# 
# Write a program that asks the user to enter the number of hours worked and the hourly rate. 
# 
# It should then display the gross pay, the amount of taxes, and the net pay. 
# 
# Assume that the employee is single and that his taxable income is 10,000. 
# 
# Use the following tax rates:
# 
# |Rate|Taxable Income|Tax Owed|
# |---|---|---|
# |10%|\$0 to \$8,000|\$0|
```

Transformers are still far from being robust. However, language model parameter counts have been increasing ten-fold every year, and it is an open question when or if scale will fix these issues. 

### Conclusion

We have seen that using code as an intermediate logical form is a general and effective method for natural language reasoning. This approach can be conceptualized as a neurosymbolic method, where a neural network leverages its natural language understanding and commonsense knowledge to decompose a problem into discrete operations that are executed by a rule-based programming language. 

What is particularly promising about applying code-generation in general purpose languages is the generality of the method. With the help of external libraries, programming languages provide a scaffolding to express any kind of logical, mathematical, or algorithmic task.
