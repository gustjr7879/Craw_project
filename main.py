import pandas as pd
from aws_craw import CModels, DataPreprocessing
from config import get_args
from model import *
import gensim
import time
import sys
if __name__ == '__main__':
    args = get_args()
    pre_data = args.predata
    pre_model = args.premodel
    vec_size = args.model_vec_size
    win_size = args.model_win_size
    neg_size = args.model_neg_size
    min_size = args.model_min_size
    search_name = args.search_name
    search_skill = args.search_skill
    prompt_mode = args.prompt_mode
    if pre_data == False:
        # 크롤러 실행
        wanted = CModels.wanted_craw()
        rallit = CModels.rallit_craw()
        programmers = CModels.prog_craw()
        #통합 데이터 제작
        concat_data = DataPreprocessing.data_concat(programmers,rallit,wanted)#.drop('Unnamed: 0',axis=1)

        #기업 정보 크롤링
        names = concat_data['기업명']
        #잘못된 기업 이름 수정
        for i in range(len(names)):
            fix_names = DataPreprocessing.name_chk(names[i])
            names[i] = fix_names

        names = list(names)

        #기업 정보 크롤링
        result = CModels.company_info(names)


        integrated_data = pd.concat([concat_data,result],axis=1)#.drop('Unnamed: 0',axis=0)

        integrated_data.to_csv('./integrated_data.csv',encoding='utf-8')
    else:
        integrated_data = pd.read_csv('./integrated_data.csv')
        integrated_data = integrated_data.drop('Unnamed: 0',axis=1)

    print(integrated_data)
    exit()

    result_list = change_data(integrated_data['기술스택'])

    if pre_model == True:
        # load W2V 16, 5, 5, 5
        model = gensim.models.Word2Vec.load('./w2v_16.model')
    else:
        # train W2V by skills
        model = gensim.models.Word2Vec(sentences=result_list,vector_size=vec_size,window=win_size,negative=neg_size,min_count=min_size)
        #print('test')
    
    
    # make company vector by skills mean
    company_vector = company_to_vector(result_list,model,vec_size)

    # add company vectors to df
    new_df = integrated_data.copy()
    new_df['features'] = company_vector

    # vectors to tsne (for draw)
    tsne = TSNE(n_components=2,metric='cosine',random_state=42)
    if prompt_mode == True:
        company_name_plotly = draw_plotly_company_names(company_vector,tsne,new_df)
        skill_name_plotly = draw_plotly_skill_names(model,tsne)
        print('안녕하세요')
        print('프롬프트 모드입니다.')
        time.sleep(.5)
        print('어떤 것을 도와드릴까요?')
        time.sleep(.5)
        print('1. AWS 직무 채용을 하는 기업들의 유사도를 보고 싶어')
        print('2. 어떤 스킬을 공부하면 어떤 기업들을 갈 수 있는지 보고 싶어')
        print('3. 스킬들의 유사도를 보고 싶어')
        print('1 or 2 or 3 으로 입력해주세요')
        input = sys.stdin.readline
        search_keyword = int(input())
        cnt = 0
        while True:
            
            if search_keyword == 1:
                name_list = new_df['기업명'].unique()
                if cnt == 0:
                    company_name_plotly.show()
                    cnt += 1
                print('유사한 기업들을 검색할 수 있습니다. 원하시는 기업을 검색해주세요')
                print(f'기업 검색가능 항목 : {name_list}')
                print('종료하시려면 컨트롤 C')
                search_keyword2 = list(str(input()).split())[0]         
                sim_company = index_search(new_df,search_keyword2,company_vector)
                time.sleep(.5)
                if len(sim_company) > 0:
                    
                    print(f'{search_keyword2}와 유사한 채용을 하고 있는 기업들은 다음과 같습니다.',sim_company)
                    time.sleep(1)
                else:
                    print('목록에 없는 기업을 검색하셨어요 !')
                    time.sleep(1)

            elif search_keyword == 2:
                if cnt == 0:
                    company_name_plotly.show()
                    cnt += 1
                print('검색하는 스킬을 채용하는 기업들을 알 수 있습니다. 스킬을 검색해주세요')
                print('여러가지 스킬도 검색할 수 있습니다. ex) AWS MySQL 과 같이 띄어쓰기로 구분해서 검색해주세요')
                print(f'스킬 검색가능 항목 : {model.wv.index_to_key}')
                print('종료하시려면 컨트롤 C')

                search_keyword2 = list(str(input()).split())            
                resu = make_mean(search_keyword2,company_vector,model,new_df,vec_size)
                time.sleep(.5)
                if len(resu) > 0:

                    print(f'{search_keyword2} 검색하신 스킬을 채용하는 회사는 다음과 같습니다.',resu)
                    time.sleep(1)
                else:
                    print('목록에 없는 스킬을 검색하셨어요 !')
                    time.sleep(1)

            elif search_keyword == 3:
                if cnt == 0:
                    skill_name_plotly.show()
                    cnt += 1
                print('검색하는 스킬과 공고에 함께 올라오는 스킬들을 검색할 수 있습니다')
                print(f'스킬 검색가능 항목 : {model.wv.index_to_key}')
                print('종료하시려면 컨트롤 C')

                search_keyword2 = list(str(input()).split())[0]            
                resu = skill_search(search_keyword2,model)
                time.sleep(.5)
                if len(resu) > 0:

                    print(f'{search_keyword2} 검색하신 스킬과 유사한 스킬은 다음과 같습니다..',resu)
                    time.sleep(1)
                else:
                    print('목록에 없는 스킬을 검색하셨어요 !')
                    time.sleep(1)
    else:
        company_name_plotly = draw_plotly_company_names(company_vector,tsne,new_df)
        #company_name_plotly.show()
        sim_company = index_search(new_df,search_name,company_vector)
        skill_name_plotly = draw_plotly_skill_names(model,tsne)
        resu = make_mean(search_skill,company_vector,model,new_df,vec_size)
        print(sim_company)

        print(resu)
