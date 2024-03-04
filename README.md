# AWS 크롤링 프로젝트를 위한 레포지토리입니다      

=======
## 파일 설명
- main.py : 교육프로그램 추천 모델 실행할 수 있음
- aws_craw.py : 크롤러들 / Data preprocessing
- model.py : 추천 알고리즘
- config.py : 파라미터
- visual.ipynb : 시각화 코드들 모여있음 전체 전처리 파일로 부터 실행
## 코드 사용법
- premodel : pretrained된 모델 사용 여부 (default 0)
- predata : 크롤링 된 데이터 불러와서 사용 여부 (default 1)
- allpreprocess : 전체 전처리를 할 것인지 여부 (시각화 시 사용 default 0)
- model_vec_size : Word2Vec 모델 vectorsize (default 16)
- model_win_size : Word2Vec 모델 윈도우 사이즈 (default 3)
- model_neg_size : Word2Vec 모델 negative sample 사이즈 (default 5)
- model_min_size : Word2Vec 모델 최소 카운트 (default 4)
- search_name : 검색할 회사 이름 (default '한화비전')
- search_skill : 검색할 스킬들 (default ['AWS','MySQL'])
- prompt_mode : 프롬프트 모드 사용 (default 1)
```python

python main.py --premodel 0 --predata 1 --model_vec_size 16 --model_win_size 3 --model_neg_size 5 --model_min_size 4 --search_name 한화비전 --search_skill AWS MySQL

```      
         
## 프롬프트 모드 사용 시 
검색어와 함께 1,2,3 중 하나 입력해서 실행      
나오는대로 입력 시 채용 공고 기술스택이 유사한 기업들, 공고에 함께 등장하는 스킬들, 스킬들을 입력했을 때 채용하는 회사 등을 추천해준다.     
이 모델을 사용해서 어떤 기술스택들이 함께 나오는지, 어떤 기술 스택을 공부하면 어떤 기업에 갈 수 있는지, 채용 공고가 유사한 기업들은 뭐가 있는지 알 수 있다.        

## 시각화
visual.ipynb에 올라가있고 전처리 종료 후 실행하면 다 돌아감