import pandas as pd
import numpy as np
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.library.parameters import Season, SeasonType
from nba_api.stats.endpoints import playbyplayv3
import time
from tqdm import tqdm

def get_season_game_ids(season = Season.default):
    """Gets game ids from default season(latest) and rs"""
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season, season_type_nullable=SeasonType.default)
    all_games_ids = [game["GAME_ID"] for game in gamefinder.get_normalized_dict()["LeagueGameFinderResults"]]
    all_games_ids = list(set(all_games_ids))
    return all_games_ids

def fetch_pbp(game_id):
    try:
        df = playbyplayv3.PlayByPlayV3(game_id=game_id).get_data_frames()[0]
    except Exception as e:
        print(game_id, e)
        return None

    return df

def process_pbp(df):
    df = df.copy()

    last_row = df.iloc[-1] #integer locate
    max_period = last_row['period']
    df['max_period'] = max_period

    df['scoreHome'] = pd.to_numeric(df['scoreHome'], errors='coerce')
    df['scoreAway'] = pd.to_numeric(df['scoreAway'], errors='coerce')

    df['period_seconds'] = df['clock'].str.extract(r'PT(\d+)M([\d.]+)S').astype(float).apply(
        lambda r:r[0]*60+r[1],axis = 1
    )

    df['seconds_left'] = np.where(df['period']<=4,(4-df['period'])*720,(df['period']-4)*300) + df['period_seconds']
    

    df['margin'] = df['scoreHome'] - df['scoreAway']

    home_won = int(df['scoreHome'].iloc[-1] > df['scoreAway'].iloc[-1]) 
    df['home_won'] = home_won

    training_df = df.dropna(subset=['margin','seconds_left'])[['gameId','margin','seconds_left','home_won','period']]

    return training_df

def scrape_seasons(game_ids,limit = None):
    processed_dfs = []
    dfs = []
    if limit is None:
        ids_to_process = game_ids
    else:
        ids_to_process = game_ids[:limit]
    for gid in tqdm(ids_to_process):
        game_df = fetch_pbp(gid)
        if game_df is None:
            continue
        dfs.append(game_df)
        game_pbp = process_pbp(game_df)
        processed_dfs.append(game_pbp)
        # time.sleep(0.6)
    raw_pbp = pd.concat(dfs,ignore_index=True)
    wp_training_data = pd.concat(processed_dfs,ignore_index=True)
    return raw_pbp, wp_training_data


if __name__ == "__main__":
    season = "2025-26"
    game_ids = get_season_game_ids(season=season)
    raw,wp = scrape_seasons(game_ids,limit = None)
    print(f"No error {season} processed")
    raw.to_parquet(f"raw_training_data_{season}.parquet")
    wp.to_parquet(f"wp_training_data_{season}.parquet")