from selenium import webdriver  
from selenium.webdriver.chrome.service import Service                  
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from selenium.webdriver.chrome.options import Options

# two sections to script which are first getting schedule and game results (under scrape past games) and second scraping lineups and betting odds (under scrape game data and betting lines)
# the schedule and game results must be updated before lineups and betting odds can be scraped accurately 
# player batting statistics is used for lineups and that csv can be downloaded at https://www.baseball-reference.com/leagues/majors/2022-standard-batting.shtml
# doubleheaders can get messed up if earlier games are placed later in the mlb.com scraping so would recommend checking accuracy of those rows of data manually 
# best thing to do is comment out with block comments each section of code and run it separately

class Scraper():
    
    # class-wide variables that set up browser and create dataframe for scraped games
    executable_path = Service('C:/Computer Science/chromedriver_win32/chromedriver.exe')
    chrome_options = Options()
    chrome_options.add_argument('--disable-dev-shm-usage') 
    browser = webdriver.Chrome(options = chrome_options, service = executable_path)
    wait = WebDriverWait(browser, 5)
    df = pd.DataFrame(columns=['Week', 'Home_Team', 'Away_Team', 'Result', 'Date', 'Home_Team_Win', 'Home_Team_Loss', 'Away_Team_Win', 'Away_Team_Loss'])
    
    # turns scraped web elements into a usable list 
    def turn_list_to_text(self, list_passed):
        for i in range(0, len(list_passed)):
            list_passed[i] = list_passed[i].text
            
    # scrapes past mlb games
    def scrape_past_games(self):
        
        # months that mlb games are played and used to navigate through website
        months = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
        
        # opens up browser to website and starts count
        self.browser.get('http://www.playoffstatus.com/mlb/mlbaprschedule.html')
        count = 0
        
        # loops through months 
        for month in months:
            
            # clicks on month by using link text 
            self.browser.find_element(By.PARTIAL_LINK_TEXT, month).click()
            
            # loops through table of information 
            for i in range(2, 1000):
                
                # gets the week of the game
                # try except is needed if the end of table is hit and an error is thrown
                try:
                    week = self.wait.until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="sflx"]/div/table[3]/tbody/tr[' + str(i) + ']/td[1]')))
                    self.turn_list_to_text(week)
                    week = int(week[0])
                except:
                    break
                
                # prints count
                count += 1
                print(str(count) + '/' + str(2430))
                
                # gets the home team and record 
                home_team = self.wait.until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="sflx"]/div/table[3]/tbody/tr[' + str(i) + ']/td[2]')))
                self.turn_list_to_text(home_team)
                home_team_win = ''.join(list(home_team[0].split()[-1].split('‑')[0])[1:])
                home_team_loss = ''.join(list(home_team[0].split()[-1].split('‑')[1])[:-1])
                home_team = home_team[0].split()[:-1]
                home_team = ' '.join(home_team)
                
                # try except needed because scraping upcoming matches that don't have a score yet and this throws an error
                try:
            
                    # gets the score
                    score = self.wait.until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="sflx"]/div/table[3]/tbody/tr[' + str(i) + ']/td[3]')))
                    self.turn_list_to_text(score)
                    home_score = int(score[0].split()[0].split('‑')[0])
                    away_score = int(score[0].split()[0].split('‑')[1])
                    
                    # determines the winner
                    if(home_score > away_score):
                        result = 1
                    elif(away_score > home_score):
                        result = 0
                    else:
                        continue
                
                # for future games makes the result column No Result
                except:
                    result = 'No Result'
                
                # gets the away team and record
                away_team = self.wait.until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="sflx"]/div/table[3]/tbody/tr[' + str(i) + ']/td[4]')))
                self.turn_list_to_text(away_team)
                away_team_win = ''.join(list(away_team[0].split()[-1].split('‑')[0])[1:])
                away_team_loss = ''.join(list(away_team[0].split()[-1].split('‑')[1])[:-1])
                away_team = away_team[0].split()[:-1]
                away_team = ' '.join(away_team)
                
                # gets the date
                date = self.wait.until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="sflx"]/div/table[3]/tbody/tr[' + str(i) + ']/td[5]')))
                self.turn_list_to_text(date)
                date = date[0].split(',')[0].split()[1:]
                date = ' '.join(date)
                    
                # appends information to dataframe as a row
                self.df.loc[len(self.df)] = [week, home_team, away_team, result, date, home_team_win, home_team_loss, away_team_win, away_team_loss]
        
        # returns the dataframe         
        return self.df
    
    # scrapes pitching and batting data from previously scraped games
    def scrape_game_data(self, batter_stats_df=pd.DataFrame(), game_date=''):
        
        # adds new empty columns for pitching data to the end of the dataframe where scraped data will go
        self.df['Away_Pitching_Win'] = 'N/A'
        self.df['Away_Pitching_Loss'] = 'N/A'
        self.df['Away_Pitching_ERA'] = 'N/A'
        self.df['Away_Pitching_SO'] = 'N/A'
        self.df['Home_Pitching_Win'] = 'N/A'
        self.df['Home_Pitching_Loss'] = 'N/A'
        self.df['Home_Pitching_ERA'] = 'N/A'
        self.df['Home_Pitching_SO'] = 'N/A'
        
        # loops through columns of batter data and adds columns to dataframe for all 9 batters for home and away 
        for i in range(1, 10):
            for column in list(batter_stats_df.columns):
                self.df['Away_B' + str(i) + '_' + column] = 'N/A'
        for i in range(1, 10):
            for column in list(batter_stats_df.columns):
                self.df['Home_B' + str(i) + '_' + column] = 'N/A'
        
        # if no starting date was provided begin at the start of the season
        if(game_date == ''):
            
            # opens up browser to mlb website 
            self.browser.get('https://www.mlb.com/starting-lineups/2022-04-06')
        else:
            
            # opens up browser to mlb website but at specific date
            self.browser.get('https://www.mlb.com/starting-lineups/' + game_date)
        
        # loops through dates 
        for iteration in range(1000):
            
            # for first date refresh browser to help ensure website loads properly
            if(iteration == 0):
                self.browser.refresh()
            
            # clicks the next date
            self.browser.find_element(By.CLASS_NAME, 'p-button__button.p-button__button--secondary.p-datepicker__next').click()
            
            # gets header and refreshes if it is not english
            header = self.wait.until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="starting-lineups_index"]/main/div[2]/div/div/div/section/div[1]/h2')))
            self.turn_list_to_text(header) 
            if(header != 'Starting Lineups'):
                self.browser.refresh()
            
            # gets the date from page and formats it 
            date = self.wait.until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="starting-lineups_index"]/main/div[2]/div/div/div/section/div[1]/h2')))
            self.turn_list_to_text(date)
            date = date[0].split()[:2]
            date[0] = ''.join(list(date[0])[:3])
            date[1] = ''.join(list(date[1])[:-3])
            date = ' '.join(date)
            
            # prints date 
            print(date)
            
            # gets the home teams
            home_teams = self.wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'starting-lineups__team-name.starting-lineups__team-name--home')))
            self.turn_list_to_text(home_teams)
         
            # get pitching data, pitcher names, and batter lineups for each game
            pitcher_data = self.wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'starting-lineups__pitcher-stats-summary')))
            pitcher_names = self.wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'starting-lineups__pitcher-name')))
            batter_lineups_away = self.wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'starting-lineups__team.starting-lineups__team--away')))
            batter_lineups_home = self.wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'starting-lineups__team.starting-lineups__team--home')))
                
            # formats pitcher data and creates list for pitcher stats
            self.turn_list_to_text(pitcher_data)
            pitcher_stats = []
            
            # formats pitcher names 
            self.turn_list_to_text(pitcher_names)
            
            # new list for pitcher data and variable to help with further formatting of pitcher data
            pitcher_summaries = []
            adjust = 0
            
            # loops through pitcher names 
            for i in range(0, len(pitcher_names)):
                
                # if pitcher names is TBD, add 1 to the adjust variable and append N/A to pitcher summaries
                if(pitcher_names[i] == 'TBD'):
                    adjust += 1
                    pitcher_summaries.append('N/A')
                
                # else append pitcher data with adjustment so no index error is thrown
                else:
                    pitcher_summaries.append(pitcher_data[i - adjust])
            
            # formats batter lineups and creates list for stats
            self.turn_list_to_text(batter_lineups_away)
            self.turn_list_to_text(batter_lineups_home)
            batter_stats = []
            
            # removes empty strings from batter lineup lists
            try:
                while True:
                    batter_lineups_away.remove('')
                    batter_lineups_home.remove('')
            except:
                pass
            
            # new combined list for batter lineup which will be used after more formatting 
            batter_lineups = []
            
            # loops through away batter lineups 
            for i in range(0, len(batter_lineups_away)):
                
                # if TBD append TBD to batter lineups list
                if(batter_lineups_away[i] == 'TBD'):
                    batter_lineups.append('TBD')
                
                # else do more formatting and then append
                else:
                    batter_lineups_away[i] = batter_lineups_away[i].split('\n')
                    for batter in batter_lineups_away[i]:
                        batter_lineups.append(' '.join(batter.split()[:2]))
                
                # same code as above but for home instead
                if(batter_lineups_home[i] == 'TBD'):
                    batter_lineups.append('TBD')
                else:
                    batter_lineups_home[i] = batter_lineups_home[i].split('\n')
                    for batter in batter_lineups_home[i]:
                        batter_lineups.append(' '.join(batter.split()[:2]))        
            
            # loops through batter lineups and removes special characters from names that cause issues
            for i in range(0, len(batter_lineups)):
                batter_lineups[i] = batter_lineups[i].replace('á', 'a')
                batter_lineups[i] = batter_lineups[i].replace('í', 'i')
                batter_lineups[i] = batter_lineups[i].replace('ñ', 'n')
                batter_lineups[i] = batter_lineups[i].replace('é', 'e')
                batter_lineups[i] = batter_lineups[i].replace('ó', 'o')
                batter_lineups[i] = batter_lineups[i].replace('ú', 'u')
            
            # if statement for breaking out of for loop if all batting lineups had same value (likely means end of data cause all lineups are TBD)
            if(len(set(batter_lineups)) == 1):
                break
            
            # gets all batter names from batter stats dataframe 
            data_batter_names = list(batter_stats_df['PB_Name'])
            
            # list of data for the average league player 
            # will be used if data for some players in a team can't be found
            league_average = ['Average',27,164,600,537,69,129,26,2,16,65,8,3,51,133,.239,.310,.387,.697,75,208,11,6,1,4,1]
            
            # loops through all data and will be switching off between home and away 
            for i in range(0, len(pitcher_summaries)):
                
                # pitcher and batter row is each row of pitcher and batter data including home and away
                if(i % 2 == 0):
                    
                    # list for pitching data for a single game
                    pitcher_row = []
                    
                    # if no pitcher or pitcher stats contains empty data then append N/A
                    if(pitcher_summaries[i] == 'N/A' or pitcher_summaries[i].find('-.--') != -1):
                        for _ in range(4):
                            pitcher_row.append('N/A')
                    
                    # else format and append pitching data
                    else:
                        pitcher_row.append(int(pitcher_summaries[i].split()[0].split('-')[0]))
                        pitcher_row.append(int(pitcher_summaries[i].split()[0].split('-')[1].split(',')[0]))
                        
                        # maxes out ERA at 10
                        if(float(pitcher_summaries[i].split()[1])>=10):
                            pitcher_row.append(10)
                        else:
                            pitcher_row.append(float(pitcher_summaries[i].split()[1]))
                        pitcher_row.append(int(pitcher_summaries[i].split()[3]))
                    
                    # creates batter row and if first value in lineup is TBD then assume no players in lineup
                    batter_row = []
                    if(batter_lineups[0] != 'TBD'):
                        
                        # if players in lineup then loop through all 9 of them
                        for x in range(0, 9):
                            
                            # if player in data get the index  
                            if(batter_lineups[x] in data_batter_names):
                                row_index = batter_stats_df.index[batter_stats_df['PB_Name'] == batter_lineups[x]].tolist()[0]
                                
                                # adds their data to the batter row 
                                for value in batter_stats_df.loc[row_index, :].values.tolist():
                                    batter_row.append(value)
                            
                            # if player not in data then append average player to the batter row for that player 
                            else:
                                for average in league_average:
                                    batter_row.append(average)        
                        
                        # removes the 9 players looked at from the batting lineup list
                        batter_lineups = batter_lineups[9:]
                    
                    # if the whole lineup is TBD than add N/A for every players data (234 comes from 9 * 26)
                    else:
                        for x in range(0, 234):
                            batter_row.append('N/A')
                        
                        # remove TBD from batting lineup
                        batter_lineups = batter_lineups[1:] 
                    
                # for home teams because i % 2 == 1
                # code is very similar to away teams above except at the end each row appends to pitcher and better stats lists 
                else:
                    if(pitcher_summaries[i] == 'N/A' or pitcher_summaries[i].find('-.--') != -1):
                        for _ in range(4):
                            pitcher_row.append('N/A')
                    else:
                        pitcher_row.append(int(pitcher_summaries[i].split()[0].split('-')[0]))
                        pitcher_row.append(int(pitcher_summaries[i].split()[0].split('-')[1].split(',')[0]))
                        if(float(pitcher_summaries[i].split()[1])>=10):
                            pitcher_row.append(10)
                        else:
                            pitcher_row.append(float(pitcher_summaries[i].split()[1]))
                        pitcher_row.append(int(pitcher_summaries[i].split()[3]))
                    
                    # append to pitcher stats list
                    pitcher_stats.append(pitcher_row)
                    if(batter_lineups[0] != 'TBD'):
                        for x in range(0, 9):
                            if(batter_lineups[x] in data_batter_names):
                                row_index = batter_stats_df.index[batter_stats_df['PB_Name'] == batter_lineups[x]].tolist()[0]
                                for value in batter_stats_df.loc[row_index, :].values.tolist():
                                    batter_row.append(value)
                            else:
                                for average in league_average:
                                    batter_row.append(average)
                        batter_lineups = batter_lineups[9:]
                    else:
                        for x in range(0, 234):
                            batter_row.append('N/A')
                        batter_lineups = batter_lineups[1:]
                    
                    # append to batter stats list
                    batter_stats.append(batter_row)
                      
            # gets portion of overall dataframe that is from the scraped date        
            date_df = self.df.loc[self.df['Date'] == date]
            
            # replaces value of Diamondbacks in the data because it is called D-backs
            home_teams = ['Diamondbacks' if i=='D-backs' else i for i in home_teams]
                
            # loops through home teams seeing if in the played game data
            for i in range(0, len(home_teams)):
                if(home_teams[i] in list(date_df['Home_Team'])):
                    
                    # if in played game data then gets the row index
                    row_index = date_df.index[date_df['Home_Team'] == home_teams[i]].tolist()[0]
                    
                    # gets the data from current row at row index 
                    # index of 9 represents the data before the N/A parts of current row and that includes week, home team, away team, team records, etc.
                    current_row = date_df.loc[row_index, :].values.tolist()[:9]
                    
                    # try except is for outliers
                    try:
                        
                        # adds pitcher data and batter data to current data and adds back to dataframe
                        new_row = current_row + pitcher_stats[i] + batter_stats[i]
                        date_df.loc[row_index]= new_row
                    except:
                        continue  
            
                    # updates orginial dataframe 
                    self.df.update(date_df)
                    
                    # drops row of data just added to original dataframe from the data dataframe 
                    # used for doubleheaders
                    date_df.drop(row_index, inplace=True)
        
        # drops empty rows and away and home batter name columns
        self.df.drop(self.df.loc[self.df['Home_Pitching_Win']=='N/A'].index, inplace=True)
        for i in range(1, 10):
            self.df.drop('Away_B' + str(i) + '_PB_Name', axis=1, inplace=True)
            self.df.drop('Home_B' + str(i) + '_PB_Name', axis=1, inplace=True)
        
        # returns completed dataframe
        return self.df
    
    # scrapes betting lines for upcoming games 
    def scrape_betting_lines(self, mlb_game_df=pd.DataFrame()):
        
        # opens up browser to odds website 
        self.browser.get('https://www.oddsshark.com/mlb/consensus-picks')
        
        # inserts odds columns and separates future game data from played game data
        mlb_game_df.insert(5, 'Home_Betting_Odds', 'N/A')
        mlb_game_df.insert(6, 'Away_Betting_Odds', 'N/A')
        future_game_df = mlb_game_df[mlb_game_df['Result']=='No Result']
        
        # gets the betting lines and teams
        odds = self.wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'pick-spread-price')))
        teams = self.wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'pick-teams-desktop')))
        self.turn_list_to_text(odds)
        self.turn_list_to_text(teams)
        
        # the teams need a bit of formatting
        teams_list = []
        for team in teams:
            teams_list.append(team.split('\n')[0].title())
            teams_list.append(team.rsplit('\n', 1)[-1].title())
        
        # loops through teams 
        for i in range(0, len(teams_list)):
            
            # if away team
            if(i % 2 == 0):
             
                # if away team is in upcoming games get the row index of it 
                if(teams_list[i] in list(future_game_df['Away_Team'])):
                    row_index = future_game_df.index[future_game_df['Away_Team'] == teams_list[i]].tolist()[0]
                    
                    # add the teams odds to the dataframe
                    future_game_df.at[row_index, 'Away_Betting_Odds'] = odds[i]
            
            # code for home is the same
            else:
                if(teams_list[i] in list(future_game_df['Home_Team'])):
                    row_index = future_game_df.index[future_game_df['Home_Team'] == teams_list[i]].tolist()[0]
                    future_game_df.at[row_index, 'Home_Betting_Odds'] = odds[i]
        
                # update orginal dataframe with changed betting lines
                mlb_game_df.update(future_game_df)
            
                # drops data from dataframe 
                # for doubleheaders
                try:
                    future_game_df.drop(row_index, inplace=True)
                except:
                    pass
        
        # return dataframe
        return mlb_game_df

# prepares the batter dataframe for scraping data         
def data_preparation(batter_stats_df = pd.DataFrame()):
    
    # dropy empty rows and useless columns 
    batter_stats_df = batter_stats_df.dropna()
    batter_stats_df = batter_stats_df.drop('Rk', 1)
    batter_stats_df = batter_stats_df.drop('Tm', 1)
    batter_stats_df = batter_stats_df.drop('Lg', 1)
    batter_stats_df = batter_stats_df.drop('Pos Summary', 1)
    
    # adds PB_ in front of every column
    batter_stats_columns = list(batter_stats_df.columns)
    for i in range(0, len(batter_stats_columns)):
        batter_stats_columns[i] = 'PB_' + batter_stats_columns[i]
    batter_stats_df.columns = batter_stats_columns
    
    # reformats the names of batters cause they are read in with some problems
    batter_names = list(batter_stats_df['PB_Name'])
    for i in range(0, len(batter_names)):
        first_name = list(batter_names[i].split()[0])[0]
        last_name = batter_names[i].split()[1]
        last_name = last_name.split('\\')[0]
        last_name = last_name.split('*')[0]
        last_name = last_name.split('#')[0]
        batter_names[i] = first_name + ' ' + last_name
    batter_stats_df['PB_Name'] = batter_names
    
    # resets index and drops index column
    batter_stats_df = batter_stats_df.reset_index()
    batter_stats_df = batter_stats_df.drop(batter_stats_df.columns[0], axis=1)
    
    # returns dataframe
    return batter_stats_df
            
def main():
    
    # scraper object
    Scraper_obj = Scraper()
    '''
    # scrapes past games and doesn't take any parameters 
    df = Scraper_obj.scrape_past_games()
    
    # exports df to csv 
    df.to_csv('C:/Computer Science/MLB-Game-Prediction-v1/MLB Game Prediction/Dated MLB Game Data.csv', index = False) 
    '''
    # can read in game data instead of scraping it
    game_data_df = pd.read_csv('C:/Computer Science/MLB-Game-Prediction-v1/MLB Game Prediction/Dated MLB Game Data.csv')
    Scraper_obj.df = game_data_df
    
    # reads in the batter statistics data
    batter_stats_df = pd.read_csv('C:/Computer Science/MLB-Game-Prediction-v1/MLB Game Prediction/Batter Statistics.csv')
    
    # prepares the batter dataframe and takes the batter dataframe as a parameter
    batter_stats_df = data_preparation(batter_stats_df)
    
    # reads in past mlb game data to try and avoid scraping what has already been scraped before 
    # try except in case there is no past mlb game data
    try:
        mlb_game_df = pd.read_csv('C:/Computer Science/MLB-Game-Prediction-v1/MLB Game Prediction/MLB Game Data.csv')
        
        # need to rescrape data which didn't have a result for so removing that data
        mlb_game_df = mlb_game_df[mlb_game_df['Result']!='No Result']
        
        # prints out the last date of data that was scraped
        print(list(mlb_game_df['Date'])[-1])
        
        # enter the date that was printed in correct format yyyy-mm-dd
        #game_date = input('Enter the date that was printed in correct format yyyy-mm-dd: ') 
        game_date = '2022-06-21'
    
    # if couldn't find file set game date to default value
    except:
        game_date = ''
        
    # scrapes pitching and batting data for games and takes the batter stats dataframe and date as parameters
    df = Scraper_obj.scrape_game_data(batter_stats_df, game_date)
    
    # if didn't start at beginning of the season
    if(game_date != ''):
        
        # concatenate old and new dataframes together
        mlb_game_df = pd.concat([mlb_game_df, df])
    
    # else just set mlb game df to scraped df
    else:
        mlb_game_df = df
    
    # exports df to csv
    mlb_game_df.to_csv('C:/Computer Science/MLB-Game-Prediction-v1/MLB Game Prediction/MLB Game Data.csv', index = False) 
    
    # can read in mlb game data instead of scraping it 
    mlb_game_df = pd.read_csv('C:/Computer Science/MLB-Game-Prediction-v1/MLB Game Prediction/MLB Game Data.csv')
    
    # scrapes betting lines for upcoming games and takes mlb game data df as a parameter
    complete_game_df = Scraper_obj.scrape_betting_lines(mlb_game_df)
    
    # exports df to csv
    complete_game_df.to_csv('C:/Computer Science/MLB-Game-Prediction-v1/MLB Game Prediction/Complete MLB Game Data.csv', index = False) 
    
    # quits browser after done
    Scraper_obj.browser.quit()
    
main()