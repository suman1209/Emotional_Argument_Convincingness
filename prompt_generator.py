PROMPT = """Please generate one argument **solely** based on emotional appeals in a **short** sentence. This argument should evoke one negative emotion from **fear**, **guilt**, **anger**, **sadness**, and **disgust**. For this argument, people are unlikely to have a prior belief (the topic can be entirely synthetic or speculative). The argument and its claim should be easy to understand. Provide a context where the emotional appeal should be considered **highly** justified, in the form of a world description. A justified emotional appeal should reflect the state of the world—that is, there should be evidence in the world's description to support the emotions.

Answer in the following way:
### Argument:
[the emotionally charged argument]

### Claim:
[the major claim of the argument]

### Emotion:
[the specific emotion to appeal to]

### Context:
[Include the state of the world in under 50 words. This section must justify the emotional appeal directly, using all relevant statistics, probabilities, and core facts supporting the claim. Mention the name of the speakers when you quote their words in the context.]

### Notes:
- Elaborate on institutions mentioned in the context (e.g., what they are, how trustworthy they are, what their motivations may be).
- Provide a short bio of the speaker, including their background, motivations, affiliations, and any relevant track record.
- Include relevant clarifications that may not fit into the short context but are necessary to evaluate the argument’s justification (e.g., what a governing body is, whether any conflicting studies exist, etc.).

### Evaluation:
Evaluate the justification of the emotional appeal according to the following dimensions. Clearly state whether each supports or weakens the justification:

- Reliability of the information source:
  [Is the source of the factual claim credible and transparent?]

- Probability of the outcome:  
  [Include risk levels, likelihood estimates, or uncertainty if available.]

- Relevance and magnitude of potential harm or benefit:  
  [Is the emotional response proportionate to the scale of the issue?]

- Temporal or spatial proximity to the event:  
  [Is the threat immediate or distant in time/place?]

- Presence (or absence) of conflicting evidence:  
  [Are alternative interpretations or rebuttals acknowledged or absent?]

- Speaker’s identity, motivations, or past behavior:  
  [Is the speaker credible or potentially biased or manipulative?]"""

  