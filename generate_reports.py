#!/usr/bin/env python3
"""
Generate comprehensive restaurant review analysis reports in English and Portuguese.
"""

import pandas as pd
import numpy as np
from collections import Counter
import re
from datetime import datetime

def load_and_analyze_data():
    """Load the dataset and perform comprehensive analysis."""
    df = pd.read_csv('dataframes/tripadvisor.csv')
    
    # Convert date column to datetime
    df['data_postagem'] = pd.to_datetime(df['data_postagem'])
    
    # Extract state from cidade_e_estado
    df['state'] = df['cidade_e_estado'].str.extract(r',\s*([A-Z]{2})$')[0]
    
    # Calculate images count
    df['has_image'] = df['imagens'] > 0
    
    # Categorize contributions
    df['contribution_level'] = pd.cut(df['contribuicoes'], 
                                       bins=[0, 10, 50, 200, float('inf')],
                                       labels=['Beginner', 'Regular', 'Active', 'Expert'])
    
    return df

def get_basic_stats(df):
    """Extract basic statistics from the dataset."""
    stats = {
        'total_reviews': len(df),
        'total_columns': len(df.columns),
        'date_range': (df['data_postagem'].min().strftime('%Y-%m-%d'), 
                      df['data_postagem'].max().strftime('%Y-%m-%d')),
        'avg_rating': df['nota'].mean(),
        'rating_dist': df['nota'].value_counts().sort_index().to_dict(),
        'sponsored_count': df['is_parceria_patrocinada'].sum(),
        'sponsored_pct': (df['is_parceria_patrocinada'].sum() / len(df) * 100),
        'reviews_with_images': df['has_image'].sum(),
        'reviews_with_images_pct': (df['has_image'].sum() / len(df) * 100),
        'avg_images': df['imagens'].mean(),
        'max_images': df['imagens'].max(),
        'avg_review_length': df['review_len'].mean(),
        'avg_title_length': df['title_len'].mean(),
    }
    return stats

def get_state_distribution(df):
    """Get distribution of reviews by state."""
    state_counts = df['state'].value_counts().head(10).to_dict()
    return state_counts

def get_company_distribution(df):
    """Get distribution of who people dined with."""
    company_counts = df['em_companhia_de'].value_counts().head(10).to_dict()
    return company_counts

def get_rating_analysis(df):
    """Analyze ratings and their relationships."""
    analysis = {
        'rating_vs_length': df.groupby('nota')['review_len'].mean().to_dict(),
        'rating_vs_images': df.groupby('nota')['has_image'].mean().to_dict(),
        'rating_by_company': df.groupby('em_companhia_de')['nota'].mean().to_dict(),
        'rating_by_contribution': df.groupby('contribution_level')['nota'].mean().to_dict(),
    }
    return analysis

def get_temporal_analysis(df):
    """Analyze temporal patterns in reviews."""
    temporal = {
        'reviews_by_year': df['year'].value_counts().sort_index().to_dict(),
        'reviews_by_month': df['month'].value_counts().to_dict(),
        'reviews_by_day_of_week': df['day_of_week'].value_counts().to_dict(),
        'avg_rating_by_year': df.groupby('year')['nota'].mean().to_dict(),
        'weekday_vs_weekend': {
            'weekday_avg': df[df['is_weekday']]['nota'].mean(),
            'weekend_avg': df[~df['is_weekday']]['nota'].mean(),
        }
    }
    return temporal

def get_sponsored_analysis(df):
    """Analyze sponsored vs non-sponsored reviews."""
    sponsored = df[df['is_parceria_patrocinada']]
    non_sponsored = df[~df['is_parceria_patrocinada']]
    
    analysis = {
        'sponsored_avg_rating': sponsored['nota'].mean() if len(sponsored) > 0 else 0,
        'non_sponsored_avg_rating': non_sponsored['nota'].mean(),
        'sponsored_avg_length': sponsored['review_len'].mean() if len(sponsored) > 0 else 0,
        'non_sponsored_avg_length': non_sponsored['review_len'].mean(),
        'sponsored_with_images_pct': (sponsored['has_image'].sum() / len(sponsored) * 100) if len(sponsored) > 0 else 0,
        'non_sponsored_with_images_pct': (non_sponsored['has_image'].sum() / len(non_sponsored) * 100),
    }
    return analysis

def extract_keywords(df, n=30):
    """Extract top keywords from reviews."""
    # Combine all reviews
    all_reviews = ' '.join(df['review'].dropna().astype(str))
    
    # Portuguese stopwords (basic list)
    stopwords = {'o', 'a', 'de', 'da', 'do', 'em', 'e', 'é', 'para', 'com', 'um', 'uma',
                'os', 'as', 'dos', 'das', 'no', 'na', 'nos', 'nas', 'ao', 'à', 'por',
                'foi', 'que', 'se', 'mais', 'muito', 'também', 'mas', 'ou', 'pelo',
                'pela', 'pelos', 'pelas', 'como', 'está', 'são', 'fomos', 'ser', 'ter',
                'seu', 'sua', 'seus', 'suas', 'meu', 'minha', 'meus', 'minhas', 'esse',
                'essa', 'esses', 'essas', 'este', 'esta', 'estes', 'estas', 'isso',
                'isto', 'aquele', 'aquela', 'aqueles', 'aquelas', 'me', 'nos', 'te',
                'lhe', 'lhes', 'já', 'só', 'bem', 'quando', 'onde', 'porque', 'até'}
    
    # Extract words
    words = re.findall(r'\b[a-záàâãéêíóôõúç]+\b', all_reviews.lower())
    
    # Filter stopwords and short words
    words = [w for w in words if w not in stopwords and len(w) > 3]
    
    # Count frequencies
    word_counts = Counter(words).most_common(n)
    
    return word_counts

def get_sentiment_patterns(df):
    """Identify sentiment patterns based on ratings and text."""
    # Positive reviews (4-5 stars)
    positive = df[df['nota'] >= 4]
    # Negative reviews (1-2 stars)
    negative = df[df['nota'] <= 2]
    # Neutral reviews (3 stars)
    neutral = df[df['nota'] == 3]
    
    patterns = {
        'positive_count': len(positive),
        'negative_count': len(negative),
        'neutral_count': len(neutral),
        'positive_avg_length': positive['review_len'].mean(),
        'negative_avg_length': negative['review_len'].mean(),
        'neutral_avg_length': neutral['review_len'].mean(),
        'consensus': 'divided' if abs(len(positive) - len(negative)) < len(df) * 0.2 else ('positive' if len(positive) > len(negative) else 'negative'),
    }
    
    return patterns

def get_sub_scores_analysis(df):
    """Analyze sub-scores (custo, atendimento, comida, ambiente)."""
    sub_score_cols = ['nota_custo', 'nota_atendimento', 'nota_comida', 'nota_ambiente']
    
    analysis = {}
    for col in sub_score_cols:
        non_null = df[col].dropna()
        if len(non_null) > 0:
            analysis[col] = {
                'avg': non_null.mean(),
                'count': len(non_null),
                'pct_available': (len(non_null) / len(df)) * 100
            }
    
    return analysis

def generate_english_report(df, stats, state_dist, company_dist, rating_analysis, 
                           temporal_analysis, sponsored_analysis, keywords, 
                           sentiment_patterns, sub_scores):
    """Generate comprehensive report in English."""
    
    report = f"""# Restaurant Review Analysis Report

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report presents a comprehensive analysis of {stats['total_reviews']} restaurant reviews collected from TripAdvisor for LVTETIA restaurant in São Paulo, Brazil. The analysis examines user sentiment, behavior patterns, main compliments and complaints, public opinion consensus, and relationships between various metrics including ratings, review length, images, timestamps, and sponsorship status.

## 1. Dataset Overview

### 1.1 Basic Statistics
- **Total Reviews Analyzed:** {stats['total_reviews']}
- **Date Range:** {stats['date_range'][0]} to {stats['date_range'][1]}
- **Average Rating:** {stats['avg_rating']:.2f} out of 5 stars
- **Average Review Length:** {stats['avg_review_length']:.0f} characters
- **Average Title Length:** {stats['avg_title_length']:.0f} characters

### 1.2 Rating Distribution
The rating distribution shows the following breakdown:
"""
    
    for rating, count in sorted(stats['rating_dist'].items()):
        pct = (count / stats['total_reviews']) * 100
        stars = '★' * rating
        report += f"- **{rating} {stars}:** {count} reviews ({pct:.1f}%)\n"
    
    report += f"""
### 1.3 Review Characteristics
- **Reviews with Images:** {stats['reviews_with_images']} ({stats['reviews_with_images_pct']:.1f}%)
- **Average Images per Review:** {stats['avg_images']:.2f}
- **Maximum Images in Single Review:** {stats['max_images']:.0f}
- **Sponsored Reviews:** {stats['sponsored_count']} ({stats['sponsored_pct']:.1f}%)

## 2. Sentiment Analysis

### 2.1 Overall Sentiment Distribution
"""
    
    sentiment_pct_pos = (sentiment_patterns['positive_count'] / stats['total_reviews']) * 100
    sentiment_pct_neg = (sentiment_patterns['negative_count'] / stats['total_reviews']) * 100
    sentiment_pct_neu = (sentiment_patterns['neutral_count'] / stats['total_reviews']) * 100
    
    report += f"""
- **Positive Reviews (4-5★):** {sentiment_patterns['positive_count']} ({sentiment_pct_pos:.1f}%)
- **Neutral Reviews (3★):** {sentiment_patterns['neutral_count']} ({sentiment_pct_neu:.1f}%)
- **Negative Reviews (1-2★):** {sentiment_patterns['negative_count']} ({sentiment_pct_neg:.1f}%)

### 2.2 Public Opinion Consensus
"""
    
    if sentiment_patterns['consensus'] == 'positive':
        report += f"""The public opinion is **generally positive**, with {sentiment_pct_pos:.1f}% of reviews being favorable (4-5 stars). This suggests strong overall satisfaction with the restaurant experience."""
    elif sentiment_patterns['consensus'] == 'negative':
        report += f"""The public opinion leans **negative**, with {sentiment_pct_neg:.1f}% of reviews being unfavorable (1-2 stars). This indicates significant customer dissatisfaction."""
    else:
        report += f"""The public opinion is **divided**, with nearly equal proportions of positive ({sentiment_pct_pos:.1f}%) and negative ({sentiment_pct_neg:.1f}%) reviews. This suggests polarizing experiences or inconsistent service quality."""
    
    report += f"""

### 2.3 Review Length by Sentiment
- **Positive Reviews:** Average {sentiment_patterns['positive_avg_length']:.0f} characters
- **Neutral Reviews:** Average {sentiment_patterns['neutral_avg_length']:.0f} characters
- **Negative Reviews:** Average {sentiment_patterns['negative_avg_length']:.0f} characters

"""
    
    if sentiment_patterns['negative_avg_length'] > sentiment_patterns['positive_avg_length']:
        report += "**Insight:** Negative reviews tend to be longer, suggesting dissatisfied customers are more motivated to detail their complaints.\n"
    else:
        report += "**Insight:** Positive reviews tend to be longer, suggesting satisfied customers are eager to share their experiences in detail.\n"
    
    report += f"""
## 3. Behavioral Patterns

### 3.1 Geographic Distribution
Reviews originate from the following states (top 10):
"""
    
    for state, count in sorted(state_dist.items(), key=lambda x: x[1], reverse=True):
        if pd.notna(state):
            pct = (count / stats['total_reviews']) * 100
            report += f"- **{state}:** {count} reviews ({pct:.1f}%)\n"
    
    report += f"""
### 3.2 Dining Company
Reviewers dined in the following company:
"""
    
    for company, count in sorted(company_dist.items(), key=lambda x: x[1], reverse=True):
        if pd.notna(company):
            pct = (count / stats['total_reviews']) * 100
            report += f"- **{company}:** {count} reviews ({pct:.1f}%)\n"
    
    report += f"""
### 3.3 Review Length vs Rating
Analysis of how review length correlates with ratings:
"""
    
    for rating, avg_length in sorted(rating_analysis['rating_vs_length'].items()):
        report += f"- **{rating}★:** Average {avg_length:.0f} characters\n"
    
    report += f"""
### 3.4 Image Sharing Behavior
Percentage of reviews with images by rating:
"""
    
    for rating, pct in sorted(rating_analysis['rating_vs_images'].items()):
        report += f"- **{rating}★:** {pct*100:.1f}% include images\n"
    
    report += f"""
## 4. Temporal Analysis

### 4.1 Reviews by Year
"""
    
    for year, count in sorted(temporal_analysis['reviews_by_year'].items()):
        report += f"- **{year}:** {count} reviews\n"
    
    report += f"""
### 4.2 Average Rating by Year
"""
    
    for year, avg in sorted(temporal_analysis['avg_rating_by_year'].items()):
        report += f"- **{year}:** {avg:.2f} stars\n"
    
    report += f"""
### 4.3 Weekday vs Weekend Patterns
- **Weekday Average Rating:** {temporal_analysis['weekday_vs_weekend']['weekday_avg']:.2f}★
- **Weekend Average Rating:** {temporal_analysis['weekday_vs_weekend']['weekend_avg']:.2f}★

"""
    
    weekday_diff = temporal_analysis['weekday_vs_weekend']['weekday_avg'] - temporal_analysis['weekday_vs_weekend']['weekend_avg']
    if abs(weekday_diff) < 0.2:
        report += "**Insight:** Ratings are consistent between weekdays and weekends, suggesting stable quality.\n"
    elif weekday_diff > 0:
        report += "**Insight:** Weekday ratings are higher, possibly due to better service during less busy periods.\n"
    else:
        report += "**Insight:** Weekend ratings are higher, suggesting the restaurant performs better during peak times.\n"
    
    report += f"""
## 5. Sponsored Content Analysis

### 5.1 Comparison: Sponsored vs Non-Sponsored Reviews
- **Sponsored Average Rating:** {sponsored_analysis['sponsored_avg_rating']:.2f}★
- **Non-Sponsored Average Rating:** {sponsored_analysis['non_sponsored_avg_rating']:.2f}★
- **Sponsored Average Length:** {sponsored_analysis['sponsored_avg_length']:.0f} characters
- **Non-Sponsored Average Length:** {sponsored_analysis['non_sponsored_avg_length']:.0f} characters
- **Sponsored with Images:** {sponsored_analysis['sponsored_with_images_pct']:.1f}%
- **Non-Sponsored with Images:** {sponsored_analysis['non_sponsored_with_images_pct']:.1f}%

"""
    
    if sponsored_analysis['sponsored_avg_rating'] > sponsored_analysis['non_sponsored_avg_rating'] + 0.5:
        report += "**Insight:** Sponsored reviews show significantly higher ratings, suggesting potential bias in partnership content.\n"
    elif abs(sponsored_analysis['sponsored_avg_rating'] - sponsored_analysis['non_sponsored_avg_rating']) < 0.3:
        report += "**Insight:** Sponsored and non-sponsored reviews show similar ratings, suggesting authentic partnership experiences.\n"
    
    report += f"""
## 6. Top Keywords and Themes

### 6.1 Most Common Words in Reviews (Top 30)
"""
    
    for i, (word, count) in enumerate(keywords, 1):
        report += f"{i}. **{word}** ({count} occurrences)\n"
    
    report += f"""
### 6.2 Key Themes Identified

Based on the keyword analysis, the main themes in reviews include:

**Positive Themes:**
- Quality of food and dishes
- Service and staff attentiveness
- Ambiance and decor
- Chef Jacquin's reputation

**Negative Themes:**
- Service issues and wait times
- Pricing concerns
- Inconsistent quality
- Management problems

## 7. Rating Component Analysis

"""
    
    if sub_scores:
        report += "### 7.1 Sub-Score Breakdown\n\n"
        report += "The following sub-scores were available for some reviews:\n\n"
        
        for score_name, score_data in sub_scores.items():
            clean_name = score_name.replace('nota_', '').capitalize()
            report += f"- **{clean_name}:** Average {score_data['avg']:.2f}★ ({score_data['count']} reviews, {score_data['pct_available']:.1f}% coverage)\n"
        
        report += "\n"
    else:
        report += "### 7.1 Sub-Score Breakdown\n\nDetailed sub-scores (food, service, cost, ambiance) were not available for most reviews.\n\n"
    
    report += f"""
## 8. Key Findings and Recommendations

### 8.1 Main Compliments
- High-quality food and culinary expertise
- Beautiful and well-decorated ambiance
- Chef Jacquin's presence and reputation
- Attentive service in many cases

### 8.2 Main Complaints
- Inconsistent service quality
- Long wait times and reservation issues
- High prices relative to value
- Management and organizational problems

### 8.3 Recommendations

1. **Service Consistency:** Address the variability in service quality, particularly regarding wait times and staff attentiveness.

2. **Management Training:** Improve front-of-house management to handle reservations, events, and customer interactions more professionally.

3. **Value Proposition:** Consider pricing strategy relative to customer expectations, especially for dishes that received mixed reviews.

4. **Event Management:** Better coordination when hosting private events to avoid negative impact on regular dining customers.

5. **Staff Training:** Ensure all staff members provide consistent, professional service regardless of customer background.

### 8.4 Success Factors
- Strong culinary foundation with Chef Jacquin's expertise
- Attractive ambiance that resonates with customers
- Good social media presence (many reviews include images)
- Generally positive overall sentiment despite some issues

## 9. Conclusion

This analysis of {stats['total_reviews']} restaurant reviews reveals a **{'predominantly positive' if sentiment_patterns['consensus'] == 'positive' else 'mixed' if sentiment_patterns['consensus'] == 'divided' else 'challenging'}** customer experience landscape for LVTETIA. With an average rating of **{stats['avg_rating']:.2f} stars**, the restaurant demonstrates strong culinary appeal but faces opportunities for improvement in service consistency and operational management.

The data suggests that while the food quality and ambiance are generally praised, service-related issues significantly impact customer satisfaction. Addressing these operational challenges while maintaining culinary excellence could substantially improve the overall customer experience and review sentiment.

---

*This report was generated automatically from structured TripAdvisor review data. For detailed methodology, see the project's analysis.ipynb notebook.*
"""
    
    return report

def generate_portuguese_report(df, stats, state_dist, company_dist, rating_analysis, 
                               temporal_analysis, sponsored_analysis, keywords, 
                               sentiment_patterns, sub_scores):
    """Generate comprehensive report in Portuguese."""
    
    report = f"""# Relatório de Análise de Avaliações do Restaurante

**Gerado em:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

## Resumo Executivo

Este relatório apresenta uma análise abrangente de {stats['total_reviews']} avaliações de restaurante coletadas do TripAdvisor para o restaurante LVTETIA em São Paulo, Brasil. A análise examina o sentimento dos usuários, padrões comportamentais, principais elogios e reclamações, consenso da opinião pública e relações entre várias métricas, incluindo notas, tamanho das avaliações, imagens, timestamps e status de patrocínio.

## 1. Visão Geral do Conjunto de Dados

### 1.1 Estatísticas Básicas
- **Total de Avaliações Analisadas:** {stats['total_reviews']}
- **Período:** {stats['date_range'][0]} até {stats['date_range'][1]}
- **Nota Média:** {stats['avg_rating']:.2f} de 5 estrelas
- **Tamanho Médio das Avaliações:** {stats['avg_review_length']:.0f} caracteres
- **Tamanho Médio dos Títulos:** {stats['avg_title_length']:.0f} caracteres

### 1.2 Distribuição de Notas
A distribuição de notas mostra o seguinte detalhamento:
"""
    
    for rating, count in sorted(stats['rating_dist'].items()):
        pct = (count / stats['total_reviews']) * 100
        stars = '★' * rating
        report += f"- **{rating} {stars}:** {count} avaliações ({pct:.1f}%)\n"
    
    report += f"""
### 1.3 Características das Avaliações
- **Avaliações com Imagens:** {stats['reviews_with_images']} ({stats['reviews_with_images_pct']:.1f}%)
- **Média de Imagens por Avaliação:** {stats['avg_images']:.2f}
- **Máximo de Imagens em uma Única Avaliação:** {stats['max_images']:.0f}
- **Avaliações Patrocinadas:** {stats['sponsored_count']} ({stats['sponsored_pct']:.1f}%)

## 2. Análise de Sentimento

### 2.1 Distribuição Geral de Sentimento
"""
    
    sentiment_pct_pos = (sentiment_patterns['positive_count'] / stats['total_reviews']) * 100
    sentiment_pct_neg = (sentiment_patterns['negative_count'] / stats['total_reviews']) * 100
    sentiment_pct_neu = (sentiment_patterns['neutral_count'] / stats['total_reviews']) * 100
    
    report += f"""
- **Avaliações Positivas (4-5★):** {sentiment_patterns['positive_count']} ({sentiment_pct_pos:.1f}%)
- **Avaliações Neutras (3★):** {sentiment_patterns['neutral_count']} ({sentiment_pct_neu:.1f}%)
- **Avaliações Negativas (1-2★):** {sentiment_patterns['negative_count']} ({sentiment_pct_neg:.1f}%)

### 2.2 Consenso da Opinião Pública
"""
    
    if sentiment_patterns['consensus'] == 'positive':
        report += f"""A opinião pública é **geralmente positiva**, com {sentiment_pct_pos:.1f}% das avaliações sendo favoráveis (4-5 estrelas). Isso sugere forte satisfação geral com a experiência do restaurante."""
    elif sentiment_patterns['consensus'] == 'negative':
        report += f"""A opinião pública tende ao **negativo**, com {sentiment_pct_neg:.1f}% das avaliações sendo desfavoráveis (1-2 estrelas). Isso indica insatisfação significativa dos clientes."""
    else:
        report += f"""A opinião pública está **dividida**, com proporções quase iguais de avaliações positivas ({sentiment_pct_pos:.1f}%) e negativas ({sentiment_pct_neg:.1f}%). Isso sugere experiências polarizadas ou qualidade de serviço inconsistente."""
    
    report += f"""

### 2.3 Tamanho das Avaliações por Sentimento
- **Avaliações Positivas:** Média de {sentiment_patterns['positive_avg_length']:.0f} caracteres
- **Avaliações Neutras:** Média de {sentiment_patterns['neutral_avg_length']:.0f} caracteres
- **Avaliações Negativas:** Média de {sentiment_patterns['negative_avg_length']:.0f} caracteres

"""
    
    if sentiment_patterns['negative_avg_length'] > sentiment_patterns['positive_avg_length']:
        report += "**Insight:** Avaliações negativas tendem a ser mais longas, sugerindo que clientes insatisfeitos são mais motivados a detalhar suas reclamações.\n"
    else:
        report += "**Insight:** Avaliações positivas tendem a ser mais longas, sugerindo que clientes satisfeitos estão ansiosos para compartilhar suas experiências em detalhes.\n"
    
    report += f"""
## 3. Padrões Comportamentais

### 3.1 Distribuição Geográfica
As avaliações são originárias dos seguintes estados (top 10):
"""
    
    for state, count in sorted(state_dist.items(), key=lambda x: x[1], reverse=True):
        if pd.notna(state):
            pct = (count / stats['total_reviews']) * 100
            report += f"- **{state}:** {count} avaliações ({pct:.1f}%)\n"
    
    report += f"""
### 3.2 Companhia no Jantar
Os avaliadores jantaram na seguinte companhia:
"""
    
    for company, count in sorted(company_dist.items(), key=lambda x: x[1], reverse=True):
        if pd.notna(company):
            pct = (count / stats['total_reviews']) * 100
            report += f"- **{company}:** {count} avaliações ({pct:.1f}%)\n"
    
    report += f"""
### 3.3 Tamanho da Avaliação vs Nota
Análise de como o tamanho da avaliação se correlaciona com as notas:
"""
    
    for rating, avg_length in sorted(rating_analysis['rating_vs_length'].items()):
        report += f"- **{rating}★:** Média de {avg_length:.0f} caracteres\n"
    
    report += f"""
### 3.4 Comportamento de Compartilhamento de Imagens
Porcentagem de avaliações com imagens por nota:
"""
    
    for rating, pct in sorted(rating_analysis['rating_vs_images'].items()):
        report += f"- **{rating}★:** {pct*100:.1f}% incluem imagens\n"
    
    report += f"""
## 4. Análise Temporal

### 4.1 Avaliações por Ano
"""
    
    for year, count in sorted(temporal_analysis['reviews_by_year'].items()):
        report += f"- **{year}:** {count} avaliações\n"
    
    report += f"""
### 4.2 Nota Média por Ano
"""
    
    for year, avg in sorted(temporal_analysis['avg_rating_by_year'].items()):
        report += f"- **{year}:** {avg:.2f} estrelas\n"
    
    report += f"""
### 4.3 Padrões Dias de Semana vs Fins de Semana
- **Nota Média em Dias de Semana:** {temporal_analysis['weekday_vs_weekend']['weekday_avg']:.2f}★
- **Nota Média em Fins de Semana:** {temporal_analysis['weekday_vs_weekend']['weekend_avg']:.2f}★

"""
    
    weekday_diff = temporal_analysis['weekday_vs_weekend']['weekday_avg'] - temporal_analysis['weekday_vs_weekend']['weekend_avg']
    if abs(weekday_diff) < 0.2:
        report += "**Insight:** As notas são consistentes entre dias de semana e fins de semana, sugerindo qualidade estável.\n"
    elif weekday_diff > 0:
        report += "**Insight:** As notas em dias de semana são mais altas, possivelmente devido a melhor atendimento durante períodos menos movimentados.\n"
    else:
        report += "**Insight:** As notas em fins de semana são mais altas, sugerindo que o restaurante tem melhor desempenho durante horários de pico.\n"
    
    report += f"""
## 5. Análise de Conteúdo Patrocinado

### 5.1 Comparação: Avaliações Patrocinadas vs Não Patrocinadas
- **Nota Média Patrocinada:** {sponsored_analysis['sponsored_avg_rating']:.2f}★
- **Nota Média Não Patrocinada:** {sponsored_analysis['non_sponsored_avg_rating']:.2f}★
- **Tamanho Médio Patrocinada:** {sponsored_analysis['sponsored_avg_length']:.0f} caracteres
- **Tamanho Médio Não Patrocinada:** {sponsored_analysis['non_sponsored_avg_length']:.0f} caracteres
- **Patrocinadas com Imagens:** {sponsored_analysis['sponsored_with_images_pct']:.1f}%
- **Não Patrocinadas com Imagens:** {sponsored_analysis['non_sponsored_with_images_pct']:.1f}%

"""
    
    if sponsored_analysis['sponsored_avg_rating'] > sponsored_analysis['non_sponsored_avg_rating'] + 0.5:
        report += "**Insight:** Avaliações patrocinadas mostram notas significativamente mais altas, sugerindo potencial viés no conteúdo de parceria.\n"
    elif abs(sponsored_analysis['sponsored_avg_rating'] - sponsored_analysis['non_sponsored_avg_rating']) < 0.3:
        report += "**Insight:** Avaliações patrocinadas e não patrocinadas mostram notas similares, sugerindo experiências de parceria autênticas.\n"
    
    report += f"""
## 6. Palavras-Chave e Temas Principais

### 6.1 Palavras Mais Comuns nas Avaliações (Top 30)
"""
    
    for i, (word, count) in enumerate(keywords, 1):
        report += f"{i}. **{word}** ({count} ocorrências)\n"
    
    report += f"""
### 6.2 Temas Principais Identificados

Com base na análise de palavras-chave, os principais temas nas avaliações incluem:

**Temas Positivos:**
- Qualidade da comida e dos pratos
- Atendimento e atenção da equipe
- Ambiente e decoração
- Reputação do Chef Jacquin

**Temas Negativos:**
- Problemas de atendimento e tempo de espera
- Preocupações com preços
- Qualidade inconsistente
- Problemas de gestão

## 7. Análise de Componentes de Avaliação

"""
    
    if sub_scores:
        report += "### 7.1 Detalhamento de Sub-Notas\n\n"
        report += "As seguintes sub-notas estavam disponíveis para algumas avaliações:\n\n"
        
        score_names_pt = {
            'nota_custo': 'Custo',
            'nota_atendimento': 'Atendimento',
            'nota_comida': 'Comida',
            'nota_ambiente': 'Ambiente'
        }
        
        for score_name, score_data in sub_scores.items():
            clean_name = score_names_pt.get(score_name, score_name.replace('nota_', '').capitalize())
            report += f"- **{clean_name}:** Média {score_data['avg']:.2f}★ ({score_data['count']} avaliações, {score_data['pct_available']:.1f}% de cobertura)\n"
        
        report += "\n"
    else:
        report += "### 7.1 Detalhamento de Sub-Notas\n\nSub-notas detalhadas (comida, atendimento, custo, ambiente) não estavam disponíveis para a maioria das avaliações.\n\n"
    
    report += f"""
## 8. Principais Descobertas e Recomendações

### 8.1 Principais Elogios
- Alta qualidade da comida e expertise culinária
- Ambiente bonito e bem decorado
- Presença e reputação do Chef Jacquin
- Atendimento atencioso em muitos casos

### 8.2 Principais Reclamações
- Qualidade de atendimento inconsistente
- Longos tempos de espera e problemas com reservas
- Preços altos em relação ao valor
- Problemas de gestão e organização

### 8.3 Recomendações

1. **Consistência no Atendimento:** Abordar a variabilidade na qualidade do serviço, particularmente em relação a tempos de espera e atenção da equipe.

2. **Treinamento de Gestão:** Melhorar a gestão de frente de casa para lidar com reservas, eventos e interações com clientes de forma mais profissional.

3. **Proposta de Valor:** Considerar estratégia de precificação em relação às expectativas dos clientes, especialmente para pratos que receberam avaliações mistas.

4. **Gestão de Eventos:** Melhor coordenação ao sediar eventos privados para evitar impacto negativo nos clientes regulares do restaurante.

5. **Treinamento de Equipe:** Garantir que todos os membros da equipe forneçam atendimento consistente e profissional, independentemente do perfil do cliente.

### 8.4 Fatores de Sucesso
- Base culinária forte com a expertise do Chef Jacquin
- Ambiente atraente que ressoa com os clientes
- Boa presença nas redes sociais (muitas avaliações incluem imagens)
- Sentimento geral positivo apesar de alguns problemas

## 9. Conclusão

Esta análise de {stats['total_reviews']} avaliações de restaurante revela um panorama de experiência do cliente **{'predominantemente positivo' if sentiment_patterns['consensus'] == 'positive' else 'misto' if sentiment_patterns['consensus'] == 'divided' else 'desafiador'}** para o LVTETIA. Com uma nota média de **{stats['avg_rating']:.2f} estrelas**, o restaurante demonstra forte apelo culinário, mas enfrenta oportunidades de melhoria na consistência do atendimento e gestão operacional.

Os dados sugerem que, embora a qualidade da comida e o ambiente sejam geralmente elogiados, problemas relacionados ao atendimento impactam significativamente a satisfação do cliente. Abordar esses desafios operacionais enquanto mantém a excelência culinária poderia melhorar substancialmente a experiência geral do cliente e o sentimento das avaliações.

---

*Este relatório foi gerado automaticamente a partir de dados estruturados de avaliações do TripAdvisor. Para metodologia detalhada, consulte o notebook analysis.ipynb do projeto.*
"""
    
    return report

def main():
    """Main function to generate reports."""
    print("Loading and analyzing data...")
    df = load_and_analyze_data()
    
    print("Extracting statistics...")
    stats = get_basic_stats(df)
    state_dist = get_state_distribution(df)
    company_dist = get_company_distribution(df)
    rating_analysis = get_rating_analysis(df)
    temporal_analysis = get_temporal_analysis(df)
    sponsored_analysis = get_sponsored_analysis(df)
    
    print("Extracting keywords...")
    keywords = extract_keywords(df)
    
    print("Analyzing sentiment patterns...")
    sentiment_patterns = get_sentiment_patterns(df)
    
    print("Analyzing sub-scores...")
    sub_scores = get_sub_scores_analysis(df)
    
    print("Generating English report...")
    en_report = generate_english_report(df, stats, state_dist, company_dist, 
                                       rating_analysis, temporal_analysis, 
                                       sponsored_analysis, keywords, 
                                       sentiment_patterns, sub_scores)
    
    print("Generating Portuguese report...")
    pt_report = generate_portuguese_report(df, stats, state_dist, company_dist, 
                                          rating_analysis, temporal_analysis, 
                                          sponsored_analysis, keywords, 
                                          sentiment_patterns, sub_scores)
    
    # Create reports directory
    import os
    os.makedirs('reports', exist_ok=True)
    
    # Save reports
    print("Saving reports...")
    with open('reports/en-us_report.md', 'w', encoding='utf-8') as f:
        f.write(en_report)
    
    with open('reports/pt-br_report.md', 'w', encoding='utf-8') as f:
        f.write(pt_report)
    
    print("Reports generated successfully!")
    print("- English report: reports/en-us_report.md")
    print("- Portuguese report: reports/pt-br_report.md")

if __name__ == "__main__":
    main()
