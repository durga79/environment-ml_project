import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Tuple


class AirQualityDataGenerator:
    
    def __init__(self, n_samples: int = 15000, random_state: int = 42):
        self.n_samples = n_samples
        np.random.seed(random_state)
        random.seed(random_state)
    
    def generate_dataset(self) -> pd.DataFrame:
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(hours=i) for i in range(self.n_samples)]
        
        countries = ['Germany', 'France', 'Italy', 'Spain', 'Poland', 'Netherlands', 
                    'Belgium', 'Austria', 'Sweden', 'Portugal']
        cities = {
            'Germany': ['Berlin', 'Munich', 'Hamburg', 'Frankfurt'],
            'France': ['Paris', 'Lyon', 'Marseille', 'Toulouse'],
            'Italy': ['Rome', 'Milan', 'Naples', 'Turin'],
            'Spain': ['Madrid', 'Barcelona', 'Valencia', 'Seville'],
            'Poland': ['Warsaw', 'Krakow', 'Wroclaw', 'Poznan'],
            'Netherlands': ['Amsterdam', 'Rotterdam', 'The Hague', 'Utrecht'],
            'Belgium': ['Brussels', 'Antwerp', 'Ghent', 'Bruges'],
            'Austria': ['Vienna', 'Salzburg', 'Graz', 'Innsbruck'],
            'Sweden': ['Stockholm', 'Gothenburg', 'Malmo', 'Uppsala'],
            'Portugal': ['Lisbon', 'Porto', 'Braga', 'Coimbra']
        }
        
        stations = ['Urban_Traffic', 'Urban_Background', 'Suburban', 'Rural', 'Industrial']
        
        data = []
        
        for i, date in enumerate(dates):
            country = np.random.choice(countries)
            city = np.random.choice(cities[country])
            station_type = np.random.choice(stations)
            
            hour = date.hour
            month = date.month
            day_of_week = date.weekday()
            season = (month % 12 + 3) // 3
            
            temperature = 10 + 15 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 5)
            humidity = 50 + 20 * np.sin(2 * np.pi * month / 12 + np.pi) + np.random.normal(0, 10)
            humidity = np.clip(humidity, 20, 95)
            
            wind_speed = np.abs(np.random.normal(15, 8))
            
            precipitation = 0
            if np.random.random() < 0.2:
                precipitation = np.abs(np.random.exponential(2))
            
            traffic_factor = 1.0
            if station_type in ['Urban_Traffic', 'Urban_Background']:
                traffic_factor = 1.5
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                traffic_factor *= 1.3
            if day_of_week < 5:
                traffic_factor *= 1.1
            
            seasonal_factor = 1.0
            if season == 1:
                seasonal_factor = 1.3
            
            pm25_base = 15 + 10 * traffic_factor * seasonal_factor
            pm25 = pm25_base + np.random.normal(0, 5) - wind_speed * 0.3
            pm25 = np.clip(pm25, 0, 150)
            
            pm10_base = pm25 * 1.5 + 5
            pm10 = pm10_base + np.random.normal(0, 8) - wind_speed * 0.4
            pm10 = np.clip(pm10, 0, 250)
            
            no2_base = 20 + 15 * traffic_factor
            no2 = no2_base + np.random.normal(0, 8) - wind_speed * 0.5
            no2 = np.clip(no2, 0, 200)
            
            co_base = 0.3 + 0.2 * traffic_factor
            co = co_base + np.random.normal(0, 0.1) - wind_speed * 0.01
            co = np.clip(co, 0, 5)
            
            o3_base = 40 + 20 * np.sin(2 * np.pi * month / 12)
            o3 = o3_base + np.random.normal(0, 10) + wind_speed * 0.3
            o3 = np.clip(o3, 0, 180)
            
            so2_base = 5 + 5 * (1 if station_type == 'Industrial' else 0)
            so2 = so2_base + np.random.normal(0, 3)
            so2 = np.clip(so2, 0, 80)
            
            aqi = self._calculate_aqi(pm25, pm10, no2, o3, co, so2)
            aqi_category = self._get_aqi_category(aqi)
            
            data.append({
                'timestamp': date,
                'country': country,
                'city': city,
                'station_type': station_type,
                'temperature': round(temperature, 2),
                'humidity': round(humidity, 2),
                'wind_speed': round(wind_speed, 2),
                'precipitation': round(precipitation, 2),
                'pm2.5': round(pm25, 2),
                'pm10': round(pm10, 2),
                'no2': round(no2, 2),
                'co': round(co, 3),
                'o3': round(o3, 2),
                'so2': round(so2, 2),
                'aqi': round(aqi, 2),
                'aqi_category': aqi_category,
                'hour': hour,
                'day_of_week': day_of_week,
                'month': month,
                'season': season
            })
        
        df = pd.DataFrame(data)
        
        missing_mask = np.random.random(df.shape) < 0.02
        df = df.mask(missing_mask)
        
        return df
    
    def _calculate_aqi(self, pm25, pm10, no2, o3, co, so2):
        pm25_aqi = (pm25 / 35) * 100
        pm10_aqi = (pm10 / 50) * 100
        no2_aqi = (no2 / 40) * 100
        o3_aqi = (o3 / 100) * 100
        co_aqi = (co / 4) * 100
        so2_aqi = (so2 / 20) * 100
        
        return max(pm25_aqi, pm10_aqi, no2_aqi, o3_aqi, co_aqi, so2_aqi)
    
    def _get_aqi_category(self, aqi):
        if aqi <= 50:
            return 'Good'
        elif aqi <= 100:
            return 'Moderate'
        elif aqi <= 150:
            return 'Unhealthy_for_Sensitive'
        elif aqi <= 200:
            return 'Unhealthy'
        elif aqi <= 300:
            return 'Very_Unhealthy'
        else:
            return 'Hazardous'


class ClimateTextDataGenerator:
    
    def __init__(self, n_samples: int = 9500, random_state: int = 42):
        self.n_samples = n_samples
        np.random.seed(random_state)
        random.seed(random_state)
        
        self.positive_templates = [
            "The European Union has announced {policy} aimed at reducing carbon emissions by {percent}% by {year}. This initiative includes investments in {technology} and strict regulations on {sector}. Environmental experts praise the comprehensive approach to tackling climate change through innovative solutions and sustainable practices.",
            "A breakthrough in {technology} technology promises to revolutionize the renewable energy sector. Scientists at {institution} have developed a new method that increases efficiency by {percent}%. This advancement could significantly reduce our dependence on fossil fuels and accelerate the transition to clean energy sources across Europe.",
            "New regulations on {sector} emissions have been successfully implemented in {country}, resulting in a {percent}% reduction in greenhouse gases. The policy framework includes incentives for businesses adopting {technology} and penalties for non-compliance. Stakeholders report positive outcomes and improved air quality in major cities.",
            "Investment in {technology} infrastructure reaches record levels as governments commit to achieving net-zero targets by {year}. The funding will support research, development, and deployment of sustainable solutions across multiple sectors including transportation, energy, and manufacturing.",
            "Community-led initiatives in {country} demonstrate remarkable success in reducing local pollution through grassroots environmental programs. Citizens have adopted {technology} practices and sustainable lifestyle changes, inspiring similar movements across Europe. Local authorities credit education and community engagement for the positive transformation."
        ]
        
        self.negative_templates = [
            "Despite promises, {country} fails to meet {year} carbon reduction targets as {sector} emissions continue to rise by {percent}%. Critics argue that insufficient investment in {technology} and weak enforcement of environmental regulations undermine climate goals. The lack of political will threatens the region's commitment to international climate agreements.",
            "New study reveals that current {policy} measures are inadequate to address the climate crisis, with experts warning of catastrophic consequences if immediate action is not taken. The report highlights deficiencies in {sector} regulation and calls for urgent reforms to prevent irreversible environmental damage.",
            "Controversy surrounds the proposed {technology} project as environmental groups raise concerns about ecological impact and long-term sustainability. The plan has faced significant opposition from local communities and scientists who question the effectiveness of the approach in achieving meaningful emissions reductions.",
            "Air quality in major European cities deteriorates as {sector} pollution increases by {percent}%. Health officials warn of rising respiratory illnesses linked to poor air quality, particularly affecting vulnerable populations. Calls for stricter regulations and immediate intervention grow louder amid public health concerns.",
            "Climate negotiations stall as nations fail to reach consensus on {policy} implementation. The breakdown highlights divisions between industrial interests and environmental priorities, casting doubt on the feasibility of meeting {year} targets. Activists express frustration over the slow pace of meaningful climate action."
        ]
        
        self.neutral_templates = [
            "Research team at {institution} publishes findings on the relationship between {sector} emissions and air quality. The study analyzed data from {country} over a period of ten years, providing insights into pollution patterns and meteorological factors. Further investigation is needed to establish causal relationships.",
            "Conference on {policy} brings together policymakers, scientists, and industry leaders to discuss {technology} strategies. Participants exchanged perspectives on challenges and opportunities in the transition to sustainable energy systems. The event facilitated dialogue but did not result in concrete commitments.",
            "Survey shows mixed public opinion on {technology} adoption, with {percent}% of respondents supporting the initiative while others express concerns about costs and implementation. The findings reflect the complexity of balancing environmental priorities with economic considerations in modern society.",
            "Annual report on {sector} trends indicates modest changes in emission levels across Europe. While some countries show progress in reducing greenhouse gases, others struggle with industrial growth and energy demands. The data suggests varied approaches to environmental policy with differing outcomes.",
            "Technical analysis examines the feasibility of {technology} deployment in {country}. Engineers assess infrastructure requirements, costs, and potential benefits. The report provides recommendations for decision-makers considering investments in renewable energy and sustainable development projects."
        ]
        
        self.policies = ['carbon tax legislation', 'renewable energy mandate', 'emissions trading scheme', 
                        'green building standards', 'clean air act', 'sustainable transport policy',
                        'circular economy framework', 'climate adaptation strategy']
        
        self.technologies = ['solar energy', 'wind power', 'electric vehicles', 'hydrogen fuel cells',
                           'carbon capture', 'battery storage', 'smart grid systems', 'bioenergy',
                           'geothermal energy', 'energy efficiency']
        
        self.sectors = ['transportation', 'manufacturing', 'agriculture', 'energy production',
                       'construction', 'aviation', 'shipping', 'industrial processes']
        
        self.countries = ['Germany', 'France', 'Italy', 'Spain', 'Poland', 'Netherlands',
                         'Sweden', 'Belgium', 'Austria', 'Portugal', 'Denmark', 'Finland']
        
        self.institutions = ['Technical University of Munich', 'Imperial College London',
                           'ETH Zurich', 'University of Copenhagen', 'Delft University',
                           'KTH Royal Institute of Technology', 'EPFL', 'University of Vienna']
    
    def generate_dataset(self) -> pd.DataFrame:
        data = []
        
        n_positive = int(self.n_samples * 0.35)
        n_negative = int(self.n_samples * 0.35)
        n_neutral = self.n_samples - n_positive - n_negative
        
        sentiments = (['Positive'] * n_positive + 
                     ['Negative'] * n_negative + 
                     ['Neutral'] * n_neutral)
        random.shuffle(sentiments)
        
        start_date = datetime(2020, 1, 1)
        
        for i, sentiment in enumerate(sentiments):
            if sentiment == 'Positive':
                template = random.choice(self.positive_templates)
            elif sentiment == 'Negative':
                template = random.choice(self.negative_templates)
            else:
                template = random.choice(self.neutral_templates)
            
            text = template.format(
                policy=random.choice(self.policies),
                technology=random.choice(self.technologies),
                sector=random.choice(self.sectors),
                country=random.choice(self.countries),
                institution=random.choice(self.institutions),
                percent=random.randint(10, 80),
                year=random.choice([2025, 2030, 2035, 2040, 2050])
            )
            
            publication_date = start_date + timedelta(days=random.randint(0, 1800))
            
            source_type = random.choice(['Policy_Document', 'News_Article', 'Research_Paper',
                                        'Government_Report', 'Press_Release'])
            
            urgency = 'High' if sentiment == 'Negative' else ('Medium' if sentiment == 'Neutral' else 'Low')
            
            impact_score = random.uniform(3, 5) if sentiment == 'Positive' else (
                random.uniform(1, 3) if sentiment == 'Negative' else random.uniform(2.5, 3.5)
            )
            
            data.append({
                'document_id': f'DOC_{i+1:05d}',
                'publication_date': publication_date,
                'text': text,
                'sentiment': sentiment,
                'source_type': source_type,
                'urgency': urgency,
                'impact_score': round(impact_score, 2),
                'word_count': len(text.split())
            })
        
        df = pd.DataFrame(data)
        
        missing_indices = random.sample(range(len(df)), int(len(df) * 0.01))
        for idx in missing_indices:
            col = random.choice(['source_type', 'urgency'])
            df.at[idx, col] = None
        
        return df


def create_integrated_dataset(air_quality_df: pd.DataFrame, 
                              text_df: pd.DataFrame) -> pd.DataFrame:
    air_quality_daily = air_quality_df.copy()
    air_quality_daily['date'] = pd.to_datetime(air_quality_daily['timestamp']).dt.date
    
    daily_agg = air_quality_daily.groupby(['date', 'country']).agg({
        'pm2.5': 'mean',
        'pm10': 'mean',
        'no2': 'mean',
        'aqi': 'mean',
        'aqi_category': lambda x: x.mode()[0] if not x.mode().empty else 'Moderate'
    }).reset_index()
    
    text_daily = text_df.copy()
    text_daily['date'] = pd.to_datetime(text_daily['publication_date']).dt.date
    
    sentiment_counts = text_daily.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
    sentiment_counts = sentiment_counts.reset_index()
    
    integrated = []
    
    for _, row in daily_agg.iterrows():
        date = row['date']
        country = row['country']
        
        matching_texts = text_daily[
            (text_daily['date'] >= date - timedelta(days=3)) &
            (text_daily['date'] <= date + timedelta(days=3))
        ]
        
        if len(matching_texts) > 0:
            sentiment_score = (
                len(matching_texts[matching_texts['sentiment'] == 'Positive']) * 1 +
                len(matching_texts[matching_texts['sentiment'] == 'Neutral']) * 0 +
                len(matching_texts[matching_texts['sentiment'] == 'Negative']) * -1
            ) / len(matching_texts)
            
            avg_impact = matching_texts['impact_score'].mean()
            
            integrated.append({
                'date': date,
                'country': country,
                'avg_pm25': round(row['pm2.5'], 2),
                'avg_pm10': round(row['pm10'], 2),
                'avg_no2': round(row['no2'], 2),
                'avg_aqi': round(row['aqi'], 2),
                'aqi_category': row['aqi_category'],
                'sentiment_score': round(sentiment_score, 3),
                'avg_impact_score': round(avg_impact, 2),
                'num_documents': len(matching_texts)
            })
    
    integrated_df = pd.DataFrame(integrated)
    
    return integrated_df.head(8000)


