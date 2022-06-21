import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle
import numpy as np
from skfeature.function.similarity_based import fisher_score
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
import operator
from sklearn.model_selection import KFold
import warnings
import sys
import os

# this script can be split into two primary sections and that is feature selection (using test_feature_selection method) and then training and predicting on mlb game data
# mlb game data from the mlb game data scraper is read in along with team batting and pitching statistics which can be found at https://www.baseball-reference.com/leagues/majors/2022.shtml
# the read in data is then formatted before being fit to various model 
# for feature selection which is just one call to a method (commented out right now in main) an important thing to remember is to remove feature_lists[i] from the data preparation method call above it 
# for feature selection i would comment out the code below the method call and when done comment out feature selection method call and of course put feature_lists[i] back in
# if not doing feature selection simply use preset features 
# for ML_training you can also choose for it to print out a 80/20 accuracy or a kfold one (currently commented out) just comment out the other ones code

# global removal of convergence warning
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

class ML():
    
    # constructor
    def __init__(self, df=pd.DataFrame()):
        self.df = df
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        
    # splits into training and testing datasets 
    def train_test_split(self, test_size=.2, target_column_name=''): 
        X = self.df.drop([target_column_name], axis=1)
        y = self.df[target_column_name]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = test_size, random_state = 22)
    
    # analysis using ML models on data (for model accuracy testing)
    def ML_analysis(self):
        
        # creates a model pipeline and appends various models for testing
        model_pipeline = []
        model_pipeline.append(LogisticRegression())
        model_pipeline.append(RandomForestClassifier())
        model_pipeline.append(SVC())
        model_pipeline.append(KNeighborsClassifier())
        model_pipeline.append(GaussianNB())
        
        # lists for output
        model_list = ['Logistic Regression', 'Random Forest', 'SVC', 'KNN', 'Naive Bayes']
        acc_list = []
        cm_list = []
        
        # loops through models
        for model in model_pipeline:
            model.fit(self.X_train, self.y_train) # fit model
            y_pred = model.predict(self.X_test) # predict using model
            
            # metrics for accuracy
            acc_list.append(model.score(self.X_test, self.y_test))
            cm_list.append(confusion_matrix(self.y_test, y_pred)) 
        '''
        # confusion matrix code and can be viewed in Plots
        
        fig = plt.figure(figsize = (18,10))
        for i in range(len(cm_list)):
            cm = cm_list[i]
            model = model_list[i]
            sub = fig.add_subplot(2, 3, i+1).set_title(model)
            cm_plot = sns.heatmap(cm, annot=True, cmap='Blues_r')
            cm_plot.set_xlabel('Predicted Values')
            cm_plot.set_ylabel('Actual Values')
        '''
        # dataframe of accuracy results     
        results_df = pd.DataFrame({'Model': model_list, 'Accuracy': acc_list})
        
        # return results dataframe
        return results_df
    
    # training one or many ML models
    def ML_training(self, model_to_use='Logistic_Regression'):
        
        # if passing in model to use then determine which model to train
        if(model_to_use == 'Logistic_Regression'):
            model = LogisticRegression()
        if(model_to_use == 'RF'):
            model = RandomForestClassifier()
        if(model_to_use == 'SVC'):
            model = SVC(probability=True)
        if(model_to_use == 'NB'):
            model = GaussianNB()
        
        # can either do train test split or kfold just comment out code not being used
        # with kfold can also see proba plots
        
        # splits into training and testing data
        self.train_test_split(.2, 'Result')
        
        # fits model to data
        model.fit(self.X_train, self.y_train)
        
        # scores testing and training data
        score = model.score(self.X_test, self.y_test)
        train_score = model.score(self.X_train, self.y_train)
        
        # prints accuracy for testing and training 
        print(pd.DataFrame({'Model': [model_to_use], 'Test Accuracy': score, 'Train Accuracy': train_score}), end='\n\n')
        '''
        # splits data into X and y
        X = np.array(self.df.drop(['Result'], axis=1))
        y = np.array(self.df['Result'])
        
        # lists for model training and testing accuracy metrics, accuracy vs probability plot, and probability vs samples plot
        train_accuracy_list = []
        test_accuracy_list = []
        probabilities = []
        accuracies = []
        probability_samples = []
        
        # creates and loops through kfold training and testing data sets
        kfold = KFold(n_splits=3, random_state=22, shuffle=True)
        for train_index, test_index in kfold.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        
            # fits the model to the training data
            model.fit(X_train, y_train)
    
            # metric for testing data accuracy
            test_accuracy_list.append(model.score(X_test, y_test))
            
            # metric for training data accuracy
            train_accuracy_list.append(model.score(X_train, y_train))
        
            # gets probability scores for model predictions 
            y_pred_prob = model.predict_proba(X_test)
            
            # loops through intervals (currently 20) for probability
            for i in range(0, 21):
                
                # gets each probability
                probability = .5 + .025 * i
                
                # list of indexes that don't meet probability will be deleted from X_test
                X_test_delete_list = []
                
                # loops through probabilities
                for i in range(0, len(y_pred_prob)):
                    
                    # if probability not in interval add to delete list
                    if(max(y_pred_prob[i]) < probability):
                        X_test_delete_list.append(i)
                        
                # delete samples from X and y test
                X_test_copy = np.delete(X_test, X_test_delete_list, axis=0) 
                y_test_copy = np.delete(y_test, X_test_delete_list, axis=0) 
                
                # try except is in case of 0 samples in interval
                try:
                    
                    # add accuracy metric, probabilities, and samples to lists
                    accuracies.append(model.score(X_test_copy, y_test_copy))
                    probabilities.append(probability)
                    probability_samples.append(len(X_test_copy))
                except:
                    continue
            
        # prints average of model accuracy for training and testing accuracy 
        print(pd.DataFrame({'Model': [model_to_use], 'Average Test Accuracy': [sum(test_accuracy_list)/len(test_accuracy_list)], 'Average Train Accuracy': [sum(train_accuracy_list)/len(train_accuracy_list)]}), end='\n\n')
        
        # can print dataframe of probabilities, accuracies, and samples that are being plotted
        #print(pd.DataFrame({'Probability': probabilities, 'Test Accuracy': accuracies, 'Samples': probability_samples}), end='\n\n')    
            
        # plots scatter and line of best fit for accuracy vs probability and sample vs probability 
        plt.figure(figsize=(6, 3))
        plt.ylim(0, 1)
        plt.xlim(.5, 1)
        plt.ylabel("Accuracy")
        plt.xlabel("Probability")
        x = np.array(probabilities)
        y = np.array(accuracies)
        a, b = np.polyfit(x, y, 1)
        plt.scatter(x, y)
        plt.plot(x, a*x+b)        
        plt.suptitle('Model Accuracy vs Probability')
        plt.figure(figsize=(6, 3))
        plt.xlim(.5, 1)
        plt.ylabel("Samples")
        plt.xlabel("Probability")
        x = np.array(probabilities)
        y = np.array(probability_samples)
        a, b = np.polyfit(x, y, 1)
        plt.scatter(x, y)
        plt.plot(x, a*x+b)  
        plt.suptitle('Model Sample vs Probability')
        '''
        # fits the model before returning it
        X = np.array(self.df.drop(['Result'], axis=1))
        y = np.array(self.df['Result'])
        model.fit(X, y)
        
        # returns the trained model
        return model
        
    # prediction using ML on future game data 
    def ML_prediction(self, model='', future_game_df_copy=pd.DataFrame()):
        
        # gets the predictions and probabilities as lists using inputed model
        y_pred = model.predict(self.df.iloc[:, 0:-1])
        y_pred_prob = model.predict_proba(self.df.iloc[:, 0:-1])
        
        # new list for probability of losing and turns probabilities to list
        loss_probabilities = []
        y_pred_prob = y_pred_prob.tolist()
        
        # loops through list of probabilities getting winning and losing probabilities 
        for i in range(0, len(y_pred_prob)):
            y_pred_prob[i] = max(y_pred_prob[i])
            loss_probabilities.append(1-y_pred_prob[i])

        # adds predictions and win/loss probabilities to dataframe
        future_game_df_copy['Result'] = y_pred
        future_game_df_copy['Win Probability'] = y_pred_prob
        future_game_df_copy['Loss Probability'] = loss_probabilities
        
        # lists for estimated return and who to bet on for each game
        money_return_list = []
        bet_on_list = []
        
        # starts looping through predictions 
        for i in range(0, len(y_pred)):
            
            # if home team predicted to win
            if(y_pred[i] == 1):
                
                # gets home odds
                home_odds = future_game_df_copy.iloc[i]['Home_Betting_Odds']
                
                # if odds are - or + a separate calculation is done (10000/odd for - and stays the same for +)
                # then multiply by the probability of that event occuring and subtract by 100 * the probability of the other event occuring 
                if(list(str(home_odds))[0] == '-'):
                    home_odds = 10000/float(''.join(list(str(home_odds))[1:]))
                    home_return = home_odds * y_pred_prob[i] - 100 * loss_probabilities[i]
                else:
                    home_return = home_odds * y_pred_prob[i] - 100 * loss_probabilities[i]
                    
                # gets away odds and same code but starts with losing probability instead
                away_odds = future_game_df_copy.iloc[i]['Away_Betting_Odds']
                if(list(str(away_odds))[0] == '-'):
                    away_odds = 10000/float(''.join(list(str(away_odds))[1:]))
                    away_return = away_odds * loss_probabilities[i] - 100 * y_pred_prob[i]
                else:
                    away_return = away_odds * loss_probabilities[i] - 100 * y_pred_prob[i]
                
                # if home team return greater append that and bet on home 
                if(home_return > away_return):
                    money_return_list.append(home_return)
                    bet_on_list.append('Home')
                
                # else do vice versa
                else:
                    money_return_list.append(away_return)
                    bet_on_list.append('Away')
                    
            # for away team winning code is the same just switched around
            else:
                away_odds = future_game_df_copy.iloc[i]['Away_Betting_Odds']
                if(list(str(away_odds))[0] == '-'):
                    away_odds = 10000/float(''.join(list(str(away_odds))[1:]))
                    away_return = away_odds * y_pred_prob[i] - 100 * loss_probabilities[i]
                else:
                    away_return = away_odds * y_pred_prob[i] - 100 * loss_probabilities[i]
                home_odds = future_game_df_copy.iloc[i]['Home_Betting_Odds']
                if(list(str(home_odds))[0] == '-'):
                    home_odds = 10000/float(''.join(list(str(home_odds))[1:]))
                    home_return = home_odds * loss_probabilities[i] - 100 * y_pred_prob[i]
                else:
                    home_return = home_odds * loss_probabilities[i] - 100 * y_pred_prob[i]
                
                if(home_return > away_return):
                    money_return_list.append(home_return)
                    bet_on_list.append('Home')
                else:
                    money_return_list.append(away_return)
                    bet_on_list.append('Away')
        
        # adds return and bet on as columns to dataframe
        future_game_df_copy['Bet_On'] = bet_on_list
        future_game_df_copy['Return'] = money_return_list
        
        # returns dataframe         
        return future_game_df_copy
    
    # gives accuracy for baseline model that uses teams wins from their record
    def baseline_model_accuracy(self, game_data_df=pd.DataFrame()):
        
        # get lists and start counter needed for baseline model 
        home_team_wins_list = game_data_df['Home_Team_Win']
        away_team_wins_list = game_data_df['Away_Team_Win']
        result_list = game_data_df['Result']
        correct_number = 0
        
        # loops through lists checking results
        for i in range(0, len(home_team_wins_list)):
         
            # uses record to determine winner and in case of same number of wins just assumes model predicts correctly half of the time 
            if((int(home_team_wins_list[i]) > int(away_team_wins_list[i]) and int(result_list[i]) == 1) or (int(home_team_wins_list[i]) < int(away_team_wins_list[i]) and int(result_list[i]) == 0)):
                correct_number += 1
            elif(int(home_team_wins_list[i]) == int(away_team_wins_list[i])):
                correct_number += .5
                
        # returns accuracy
        return float(correct_number)/len(result_list)
                
class Feature_Eng_Sel():
    
    # can normalize dataframe
    def normalize_data(self, df = pd.DataFrame()):
        for column in list(df.columns):
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        return df
    
    # determines which n features are best correlated to target column y
    def cor_selector(self, X=pd.DataFrame(), y=pd.DataFrame(), num_feats=0):
        cor_list = []
        
        # calculate the correlation with y for each feature and replace NaN with 0
        for i in X.columns.tolist():
            cor = np.corrcoef(X[i], y)[0, 1]
            cor_list.append(cor)
        cor_list = [0 if np.isnan(i) else i for i in cor_list]
        
        # gets feature names and returns list
        cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
        return cor_feature
    
    # aids in determining features by using Fisher score
    def fisher_score(self, X=pd.DataFrame(), y=pd.DataFrame(), X_columns=[], num_feats=0):
        
        # turns X and y into numpy arrays
        X = X.to_numpy()
        y = np.array(y)
        
        # gets Fisher scores
        scores = list(fisher_score.fisher_score(X, y))
        
        # creates sorted score dict with column and score
        score_dict = dict()
        for i in range(0, len(X_columns)):
            score_dict[X_columns[i]] = scores[i]
        score_dict = sorted(score_dict.items(),key=operator.itemgetter(1),reverse=True)
        
        # gets and returns top n columns according to Fisher score
        column_list = []
        for score in score_dict:
            column_list.append(score[0])
        return column_list[:num_feats]
    
    # uses recursive feature elimination on logistic regression to find features
    def recursive_feature_elimination(self, X=pd.DataFrame(), y=pd.DataFrame(), num_feats=0):
        
        # creates model and fits it
        rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
        rfe_selector.fit(X, y)
        
        # gets support, features, and returns list of features
        rfe_support = rfe_selector.get_support()
        rfe_feature = X.loc[:,rfe_support].columns.tolist()
        return rfe_feature
    
    # uses random forest model to get best features for model
    def random_forest_selection(self, X=pd.DataFrame(), y=pd.DataFrame(), num_feats=0):
        
        # creates model and fits it
        embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
        embeded_rf_selector.fit(X, y)
        
        # gets support, features, and returns list of features
        embeded_rf_support = embeded_rf_selector.get_support()
        embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
        return embeded_rf_feature
    
    # tests all feature selection methods and all number of features to find the best accuracy
    def test_feature_selection(self, team_stats_df=pd.DataFrame(), game_data_df=pd.DataFrame(), X=pd.DataFrame(), y=pd.DataFrame(), start_num=0, max_feats=0, interval=0, model_accuracy=0):
        
        # for Naive Bayes which comes in as 3 needs to be changed to 4
        if(model_accuracy == 3):
            model_accuracy = 4
        
        # variables for testing 
        max_accuracy = 0
        feature_selection = ''
        number_features = 0
        feature_list = []
        
        # loops through numbers of features at set interval
        for number in range(start_num, max_feats + 1, interval):
            
            # gets feature list that correlation recommends at specific number of features
            cor_feature_list = self.cor_selector(X, y, number)
            
            # prepares data with that feature list
            df = data_preparation(team_stats_df, game_data_df, cor_feature_list)
            
            # creates ML object and splits df into train test split
            ML_obj = ML(df)
            ML_obj.train_test_split(test_size=.2, target_column_name='Result')
           
            # gets results for those features and the max accuracy
            # index of model_accuracy is for basing accuracy on specific model
            results_df = ML_obj.ML_analysis()
            column_max = results_df['Accuracy'][model_accuracy]
            
            # sees if accuracy is higher than current max and if so change variables
            if(column_max > max_accuracy):
                max_accuracy = column_max
                feature_selection = 'Correlation'
                number_features = number
                feature_list = cor_feature_list
            
            # all other feature selection methods are very similar to correlation 
            # so mostly just rerunning the same code above for other methods 
            fisher_feature_list = self.fisher_score(X, y, list(X.columns), number)
            df = data_preparation(team_stats_df, game_data_df, fisher_feature_list)
            ML_obj = ML(df)
            ML_obj.train_test_split(test_size=.2, target_column_name='Result')
            results_df = ML_obj.ML_analysis()
            column_max = results_df['Accuracy'][model_accuracy]
            if(column_max > max_accuracy):
                max_accuracy = column_max
                feature_selection = 'Fisher'
                number_features = number
                feature_list = fisher_feature_list
            
            # for recursive feature elimination and random forest increasing iterations can increase accuracy but increase time greatly
            for _ in range(1):
                recursive_feature_list = self.recursive_feature_elimination(X, y, number)
                df = data_preparation(team_stats_df, game_data_df, recursive_feature_list)
                ML_obj = ML(df)
                ML_obj.train_test_split(test_size=.2, target_column_name='Result')
                results_df = ML_obj.ML_analysis()
                column_max = list(results_df['Accuracy'])[model_accuracy]
                if(column_max > max_accuracy):
                    max_accuracy = column_max
                    feature_selection = 'Recursive'
                    number_features = number
                    feature_list = recursive_feature_list
                forest_feature_list = self.random_forest_selection(X, y, number)
                df = data_preparation(team_stats_df, game_data_df, forest_feature_list)
                ML_obj = ML(df)
                ML_obj.train_test_split(test_size=.2, target_column_name='Result')
                results_df = ML_obj.ML_analysis()
                column_max = results_df['Accuracy'][model_accuracy]
                if(column_max > max_accuracy):
                    max_accuracy = column_max
                    feature_selection = 'Forest'
                    number_features = number
                    feature_list = forest_feature_list
                
            # prints current number and testing variables
            print(number, max_accuracy, feature_selection, number_features, feature_list)
        
        # return testing variables 
        return max_accuracy, feature_selection, number_features, feature_list
                
# some simple exploratory analysis
def exploratory_analysis(df = pd.DataFrame()):  
    
    # prints info about columns inlcuding data type and non null count for columns
    df.info()
    
    # creates a correlation matrix that can be found in plots
    correlation = df.corr()
    sns.heatmap(correlation, cmap = "GnBu", annot = True)

# preparing the data for machine learning
def data_preparation(team_stats_df=pd.DataFrame(), game_data_df=pd.DataFrame(), combined_df_list = []):
    
    # manually resetting the team stats index because it includes clubs cities and the game data doesn't have that
    team_stats_df.index = ['Diamondbacks', 'Braves', 'Orioles',
       'Red Sox', 'Cubs', 'White Sox',
       'Reds', 'Guardians', 'Rockies',
       'Tigers', 'Astros', 'Royals',
       'Angels', 'Dodgers', 'Marlins',
       'Brewers', 'Twins', 'Mets',
       'Yankees', 'Athletics', 'Phillies',
       'Pirates', 'Padres', 'Mariners',
       'Giants', 'Cardinals', 'Rays',
       'Rangers', 'Blue Jays', 'Nationals',
       'League Average']
    
    # gets all the home teams and creates the home columns list
    home_teams = list(game_data_df['Home_Team'])
    home_columns = []
    
    # loops through the columns of team stats adding Home_ to the front of each column name
    for column in list(team_stats_df.columns):
        home_columns.append('Home_' + column)
    
    # creates the dataframe for all home team data 
    home_team_df = pd.DataFrame(columns=home_columns)
    
    # loops through home teams adding each teams data to the home team dataframe 
    for home_team in home_teams:
        home_team_list = list(team_stats_df.loc[[home_team]].iloc[0])
        home_team_df.loc[len(home_team_df)] = home_team_list
    
    # everything that was done for home needs to be done for away as well
    away_teams = list(game_data_df['Away_Team'])
    away_columns = []
    for column in list(team_stats_df.columns):
        away_columns.append('Away_' + column)
    away_team_df = pd.DataFrame(columns=away_columns)
    for away_team in away_teams:
        away_team_list = list(team_stats_df.loc[[away_team]].iloc[0])
        away_team_df.loc[len(away_team_df)] = away_team_list
    
    # creates a combined dataframe with home and away team data
    combined_df = pd.concat([home_team_df, away_team_df], axis=1, join="inner")
    
    # inserts the week column at the start of the combined dataframe
    combined_df.insert(loc=0, column='Week', value=list(game_data_df['Week']))

    # add the pitching and batting data into combined dataframe
    # the index 5 is where predictive data starts and it goes to target column at the end
    combined_df = pd.concat([combined_df, game_data_df.iloc[:, 5:-1]], axis=1, join="inner")
    
    # this is used for Pearson correlation, Fisher score, recursive feature elimination, and random forest selection testing
    # keeps columns in dataframe from list that is copied from testing
    if(len(combined_df_list)!=0):
        combined_df = combined_df[combined_df.columns.intersection(combined_df_list)]
    
    # adds results column into the combined dataframe
    # try except is needed because converting all values to integer for training but for predicting this throws an error
    try:
        result_list = [int(i) for i in list(game_data_df['Result'])]
    except:
        result_list = list(game_data_df['Result'])
    
    # inserts results column into the combied dataframe at the end
    combined_df.insert(loc=len(combined_df.columns), column='Result', value=result_list)

    # returns the combined dataframe
    return combined_df

# saves ML model
def save_ML_model(model = '', file = ''):
    pickle.dump(model, open(file, 'wb'))

# loads and returns a ML model 
def load_ML_model(file = ''):
    model = pickle.load(open(file, 'rb'))
    return model  
   
def main():
    
    # reads in the team stats which can be downloaded and converted to csv manually at https://www.baseball-reference.com/leagues/majors/2022.shtml
    team_batting_stats_df = pd.read_csv('C:/Computer Science/MLB Game Prediction/Team Batting Statistics.csv', index_col=0)
    team_pitching_stats_df = pd.read_csv('C:/Computer Science/MLB Game Prediction/Team Pitching Statistics.csv', index_col=0)
    
    # adds TB_ in front of team batting columns and TP_ in front of team pitching columns
    new_batting_columns = []
    for batting_column in list(team_batting_stats_df.columns):
        new_batting_columns.append('TB_' + batting_column)
    team_batting_stats_df.columns = new_batting_columns
    new_pitching_columns = []
    for pitching_column in list(team_pitching_stats_df.columns):
        new_pitching_columns.append('TP_' + pitching_column)
    team_pitching_stats_df.columns = new_pitching_columns
    
    # adds together batting and pitching statistics
    team_stats_df = pd.concat([team_batting_stats_df, team_pitching_stats_df], axis=1)
    
    # gets the game data
    game_data_df = pd.read_csv('C:/Computer Science/MLB Game Prediction/Complete MLB Game Data.csv')

    # drops the date column and moves the result column to the end
    game_data_df = game_data_df.drop(columns=['Date'])
    results_column = game_data_df.pop("Result")
    game_data_df.insert(len(game_data_df.columns), "Result", results_column)
    
    # separates future game data from game data
    future_game_df = game_data_df[game_data_df['Result']=='No Result']
    
    # makes a copy of future game data that will be exported because its teams won't have been one hot encoded 
    future_game_df_copy = future_game_df.copy()
    
    # separates played game data from game data
    game_data_df = game_data_df[game_data_df['Result']!='No Result']
    
    # drops the betting columns from played upcoming games 
    # its not used in model prediction only for viewing afterwards
    game_data_df = game_data_df.drop(columns=['Home_Betting_Odds', 'Away_Betting_Odds'])
 
    # drops empty rows, resets index, and drops index column that is created for all 3 dataframes
    game_data_df = game_data_df.dropna()
    game_data_df = game_data_df.reset_index()
    game_data_df = game_data_df.drop(columns=['index'])
    future_game_df = future_game_df.dropna()
    future_game_df = future_game_df.reset_index()
    future_game_df = future_game_df.drop(columns=['index'])
    future_game_df_copy = future_game_df_copy.dropna()
    future_game_df_copy = future_game_df_copy.reset_index()
    future_game_df_copy = future_game_df_copy.drop(columns=['index'])
    
    # drops the betting columns from upcoming games 
    future_game_df = future_game_df.drop(columns=['Home_Betting_Odds', 'Away_Betting_Odds'])
    
    # drops columns after index 9 for future game df copy because that's where predictive data begins so don't care about that when exporting 
    future_game_df_copy.drop(future_game_df_copy.iloc[:, 9:-1], inplace = True, axis = 1)
    
    # lists for testing multiple models 
    # enter names for models and respective features
    model_to_use_list = ['Logistic_Regression', 'RF', 'SVC', 'NB']
    feature_lists = [['Away_TP_W-L%', 'Away_Pitching_Win', 'Away_Pitching_Loss', 'Away_Pitching_ERA', 'Away_Pitching_SO', 'Home_Pitching_Win', 'Home_Pitching_Loss', 'Home_Pitching_ERA', 'Home_Pitching_SO', 'Away_B6_PB_BA', 'Away_B7_PB_SLG', 'Away_B7_PB_OPS+', 'Away_B8_PB_BA', 'Away_B8_PB_OPS+', 'Home_B6_PB_OBP', 'Home_B7_PB_SLG', 'Home_B8_PB_AB', 'Home_B8_PB_SLG', 'Home_B8_PB_OPS+', 'Home_B8_PB_TB'],
                     ['Home_TP_RA/G', 'Home_TP_W', 'Home_TP_W-L%', 'Away_Pitching_Win', 'Away_Pitching_Loss', 'Away_Pitching_ERA', 'Away_Pitching_SO', 'Home_Pitching_Win', 'Home_Pitching_Loss', 'Home_Pitching_ERA', 'Home_Pitching_SO', 'Away_B7_PB_R', 'Away_B7_PB_OBP', 'Away_B7_PB_SLG', 'Away_B8_PB_BA', 'Home_B6_PB_BA', 'Home_B6_PB_OPS+', 'Home_B7_PB_AB', 'Home_B7_PB_OPS', 'Home_B8_PB_G', 'Home_B8_PB_PA', 'Home_B8_PB_AB', 'Home_B8_PB_SLG'],
                     ['Home_TP_IBB', 'Away_Pitching_Win', 'Away_Pitching_Loss', 'Away_Pitching_ERA', 'Home_Pitching_Win', 'Home_Pitching_Loss', 'Home_Pitching_ERA', 'Away_B3_PB_SB', 'Away_B5_PB_HBP', 'Away_B6_PB_HBP', 'Away_B7_PB_HBP', 'Away_B8_PB_HR', 'Home_B1_PB_Age', 'Home_B1_PB_CS', 'Home_B1_PB_HBP', 'Home_B3_PB_HBP', 'Home_B5_PB_SF', 'Home_B6_PB_GDP', 'Home_B7_PB_Age', 'Home_B7_PB_3B', 'Home_B7_PB_CS', 'Home_B7_PB_HBP', 'Home_B7_PB_SF', 'Home_B8_PB_GDP', 'Home_B8_PB_HBP', 'Home_B8_PB_SF'],
                     ['Home_TP_IBB', 'Away_Pitching_Win', 'Away_Pitching_Loss', 'Away_Pitching_ERA', 'Home_Pitching_Win', 'Home_Pitching_Loss', 'Home_Pitching_ERA', 'Away_B6_PB_HBP', 'Away_B7_PB_HBP', 'Home_B5_PB_SF', 'Home_B7_PB_3B', 'Home_B7_PB_CS', 'Home_B7_PB_HBP', 'Home_B7_PB_SF', 'Home_B8_PB_SF']]
    
    # loops through models 
    for i in range(0, len(model_to_use_list)):
    
        # prepares data for ML and will take the parameters team statistics, game data or future game data, and a list of feature columns to keep 
        # taking game data so going to be training model(s)
        # remove feature_lists[i] if doing feature selection
        df = data_preparation(team_stats_df, game_data_df, feature_lists[i])
        
        # feature engineering and selection object
        #Feature_Eng_Sel_obj = Feature_Eng_Sel()
        
        # testing for all feature selection methods
        # needs team_stats_df, game_data_df, X, y, start feature number, max feature number, interval, and the respective model on as parameters 
        #print(Feature_Eng_Sel_obj.test_feature_selection(team_stats_df, game_data_df, df.iloc[:, 0:-1], df['Result'], 20, 30, 1, i))
        
        # creates a ML object
        ML_obj = ML(df)
        
        # train test split that takes test size and target column name as parameters 
        ML_obj.train_test_split(test_size=.2, target_column_name='Result')
        
        # tests multiple ML model accuracies and prints them out and takes no parameters
        print(ML_obj.ML_analysis(), end='\n\n')
        
        # prints out a baseline model accuracy and takes no parameters 
        # the baseline model uses wins from team record to determine winners and losers
        print(str(ML_obj.baseline_model_accuracy(game_data_df)*100) + '% baseline model accuracy', end='\n\n')
        
        # training a model that takes model to use as a parameter and returns the trained model
        # prints the accuracy of trained model and plots accuracy vs probability and sample vs probability
        model = ML_obj.ML_training(model_to_use_list[i])
        
        # saves a ML model and takes a model and file as parameters
        save_ML_model(model, 'C:/Computer Science/MLB Game Prediction/Trained MLB Model.pkl')
        
        # loads and returns a saved model and takes a file as a parameter
        #model = load_ML_model('C:/Computer Science/MLB Game Prediction/Trained MLB Model.pkl')
        
        # data preparation again and this time taking future game data so going to be predicting data
        pred_df = data_preparation(team_stats_df, future_game_df, feature_lists[i])
        
        # changes ML object dataframe to pred_df 
        ML_obj.df = pred_df
        
        # predicts data and takes a model and future game data copy as parameters while returning a prediction dataframe for exporting
        prediction_df = ML_obj.ML_prediction(model, future_game_df_copy)
        
        # if first time through loop make new dataframe
        if(i == 0):
            final_df = prediction_df
            final_df.rename(columns = {'Result':model_to_use_list[i] + '_Result', 'Win Probability':model_to_use_list[i] + '_Win_Probability', 'Loss Probability':model_to_use_list[i] + '_Loss_Probability', 'Bet_On':model_to_use_list[i] + '_Bet_On', 'Return':model_to_use_list[i] + '_Return'}, inplace = True)
        
        # else add onto existing one
        else:
            final_df[model_to_use_list[i] + '_Result'] = prediction_df['Result']
            final_df[model_to_use_list[i] + '_Win_Probability'] = prediction_df['Win Probability']
            final_df[model_to_use_list[i] + '_Loss_Probability'] = prediction_df['Loss Probability']
            final_df[model_to_use_list[i] + '_Bet_On'] = prediction_df['Bet_On']
            final_df[model_to_use_list[i] + '_Return'] = prediction_df['Return']
        
    # drops extra columns from final dataframe
    final_df.drop(['Result', 'Win Probability', 'Loss Probability', 'Bet_On', 'Return'], axis = 1, inplace = True)
    
    # creates new column that averages all the models returns 
    for i in range(0, len(model_to_use_list)):
        model_to_use_list[i] = model_to_use_list[i] + '_Return'
    final_df['Average_Return'] = final_df[model_to_use_list].mean(axis=1)
    
    # exports dataframe to csv
    final_df.to_csv('C:/Computer Science/MLB Game Prediction/MLB Prediction Data.csv', index = False)
    
main()