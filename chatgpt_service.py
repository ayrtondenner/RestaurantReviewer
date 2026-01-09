from __future__ import annotations

import json
import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI


def summarize_reviews_sentiment_ptbr(
	reviews: List[str],
	*,
	model: str = "gpt-4o-mini",
) -> str:
	"""Summarize the sentiment of restaurant reviews in PT-BR.

	Sends two prompts to ChatGPT:
	- system: fixed instruction (with PT-BR constraint)
	- user: the actual list of review strings

	Args:
		reviews: List of review strings.
		model: Chat model name.

	Returns:
		A PT-BR sentiment summary.
	"""
	if not isinstance(reviews, list) or not all(isinstance(r, str) for r in reviews):
		raise TypeError("reviews must be a list[str]")

	load_dotenv()
	api_key = os.getenv("OPENAI_API_KEY")
	if not api_key:
		raise RuntimeError("OPENAI_API_KEY not found. Add it to your .env file.")

	client = OpenAI(api_key=api_key)

	system_prompt = (
		"Você vai receber uma lista de reviews de restaurantes. "
		"Escreva um resumo do sentimento do que as pessoas estão comentando online.\n\n"
		"IMPORTANTE: a entrada e a saída devem estar em português do Brasil (PT-BR)."
	)

	user_content = json.dumps(reviews, ensure_ascii=False)

	resp = client.chat.completions.create(
		model=model,
		messages=[
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_content},
		],
		temperature=0,
	)

	content = (resp.choices[0].message.content or "").strip()
	if not content:
		raise RuntimeError("Empty response from ChatGPT")
	return content
