import pandas as pd
import os
import argparse

''' 
    submission 불러오기
'''
def load_submission(args) :
    submission = pd.read_csv(os.path.join(f"/opt/ml/prediction/submission_{args.no}.csv"))
    # 분포 확인
    print(submission.pred.value_counts())

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description="Check submission arguments")
    parser.add_argument('--no', required = True, type = str, help = "input submission_no")
    
    args = parser.parse_args()
    print(args)
    load_submission(args)