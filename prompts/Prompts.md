# Basic fluency and coherence (does it stay on topic for a few sentences?)

The city council met on Tuesday evening to discuss

In a shocking finding, scientists discovered a herd of unicorns living in a remote valley in the Andes. 
*(a variant of the prompt OpenAI used in the original GPT-2 announcement, handy for comparison)*

It was a dark and stormy night, and

# Factual recall (GPT-2 small is weak here, which makes these good for seeing the limits)

The capital of France is

Water boils at a temperature of

The first person to walk on the moon was

*Use greedy decoding or a low temperature for these, so you’re testing knowledge rather than sampling noise*.

# Format and pattern following (few-shot in-context learning)

English: cat\nFrench: chat\nEnglish: dog\nFrench: chien\nEnglish: house\nFrench:

Q: What is 2 + 2?\nA: 4\nQ: What is 3 + 5?\nA: 
*(expect arithmetic to fail beyond trivial cases)*

apple, banana, cherry, date, 
*(alphabetical list continuation)*

#Structured text and code

def fibonacci(n):\n

<html>\n<head>\n<title>

{"name": "Alice", "age": 30, "city":

# Repetition and degeneration (best for comparing decoding strategies)

I don't know. I don't know. I don't know. 
*(does it break out of the loop?)*

The best thing about the weekend is 
*(run with greedy decoding and watch for loops, then with top-p 0.9 and with min-p)*

*Any of the prompts above at 200+ tokens, since coherence drift shows up over length* 

# Genre and register shifts

Dear Sir or Madam,\n\nI am writing to complain about
BREAKING NEWS:
Once upon a time, in a kingdom far away,
Ingredients:\n- 2 cups flour\n-