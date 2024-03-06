# -*- coding: utf-8 -*-
# nc *********
#
# Neighbor Centrality 계산
# 다른 노드 j로부터 노드 i가 꼽힌 순위를 rji라고 하면, 모든 다른 노드 j에 대해서
# qji = mj-rji 를 구한다.
#
# 순위를 고려하는 정도인 alpha 파라미터에 따라
#   alpha=0이면 degree centrality
#   alpha=INF이면 Nearest Neighbor Centrality
# 로 설정되게 한다.
# alpha의 권장값은 2로 설정한다.
#
#
#
# 데이터
# - linklist_raw[] : 링크 리스트 원본 [from노드, to노드, 링크강도]
# - linklist[] : 번호로 바꾸고 순서 바꾼 링크 리스트 [to노드번호, from노드번호, 링크강도]
# - nodelist[] : 리스트 순서에 맞춘 노드 목록 [이름, inw, outw]
# - nodedic[] : {노드명: 노드번호} 사전

# version 0.4
# - 결과를 node명 순으로 sort하여 출력

# version 0.3
# - input file을 닫지 않았던 오류 수정
#   linklistf.close()
#

###########################
#

import os
import sys
from time import sleep as waiting
from os.path import exists


print("------------------------------------------------------")
print("nc - Neighbor Centrality Calculation Program (v.0.3)")
print("\t\t\t\tby J. Y. Lee")
print("------------------------------------------------------\n")


# 처리할 시기의 번호를 입력 받음
period_number = int(input("처리할 시기의 번호를 입력하세요: "))


# 이전 작업 단계에서 생성된 파일 이름을 바로 사용
directory = "result"
filename = f"p{period_number}_term_cossim_edged.txt"
linklistfname = os.path.join(directory, filename)
    

# 파일에서 링크 리스트 읽기
linklist_raw = []
with open(linklistfname, 'r', encoding='utf-8') as file:
    for line in file:
        if line.strip():  # 빈 줄이 아닌 경우에만 처리
            parts = line.strip().split('\t')
            linklist_raw.append(parts)


linklist_raw = []  # 링크 리스트 원본 [from노드, to노드, 링크강도]
linklistf = open(linklistfname, 'r', encoding='utf-8')
for line in linklistf :
    if line.isspace() :   # ignore empty line
        continue

    # 줄 끝 다듬고 노드1, 노드2, 링크 강도 분할 - [from노드, to노드, 링크강도] 쌍 추출
    l = line.rstrip()
    link_info = l.split('\t')

    linklist_raw.append(link_info)
linklistf.close()

# 노드 목록과 번호 생성 (nodedic : 사전 (노드명 : 번호); nodelist : 번호별 노드명 리스트 [노드명,inw,outw] )

nodelist = [[]]
nodedic = {}
linklist = []
n_nodes = 0

# 입력 데이터의 각 링크를 순회하면서 노드 목록 완성,
# 링크 목록 테이블 linklist [to노드, from노드, 링크강도] 완성
for link in linklist_raw :

    # 앞 노드 (link[0]) 등록
    if link[0] in nodedic :
        # 사전에 등록된 노드이면 노드번호 확보
        from_node = nodedic[link[0]]

    else :
        # 미등록 노드이면 등록
        n_nodes += 1
        from_node = n_nodes
        nodedic[link[0]] = from_node   # 노드 사전에 이름과 번호 등록
        nodelist.append([link[0],0.0,0.0])   # 노드 목록에 [이름, inw, outw] 초기값 등록

    # 뒷 노드 (link[1]) 등록
    if link[1] in nodedic :
        # 사전에 등록된 노드이면 노드번호 확보
        to_node = nodedic[link[1]]
    else :
        # 미등록 노드이면 등록
        n_nodes += 1
        to_node = n_nodes
        nodedic[link[1]] = to_node   # 노드 사전에 이름과 번호 등록
        nodelist.append([link[1],0.0,0.0])   # 노드 목록에 [이름, inw, outw] 초기값 등록

    # 링크 목록 갱신
    linkw = float(link[2])
    linklist.append([to_node,from_node,linkw])

    # inw, outw 갱신
    nodelist[from_node][2] += linkw   # 내보내는 노드의 outw 합계 갱신
    nodelist[to_node][1] += linkw   # 받는 노드의 inw 합계 갱신




############

# 노드별 outlist 생성
# - linklist로부터 outlist 생성 [FromNodeID,[(value,ToNodeID),(value,ToNodeID),(value,ToNodeID)]]
# - inlist 초기화 (n_nodes 크기의 리스트의 리스트)
outlist = [[]]
for i in range(n_nodes) :
    outlist.append([])

for link in linklist :
    outlist[link[1]].append((link[2],link[0]))   # 각 FromNode의 리스트에 (값,ToNode) 튜플 저장

# 각 노드의 outlink list를 sort
for outlinks in outlist :
    if outlinks != [] :
        outlinks.sort(reverse=True)

# inlist 초기화 (n_nodes 크기의 리스트의 리스트)
inlist = [[]]
for i in range(n_nodes) :
    inlist.append([])

# outlist에서 각 노드의 outlist에 따라 inlist를 순위로 채우기
for outlinks in outlist :
    if outlinks != [] :
        # 순위를 산출하면서 inlist에 저장
        # (삭제) n_outlink = len(outlinks)
        c_r = 1
        c_value = outlinks[0][0]
        c_seq = 1
        for (outvalue,tonid) in outlinks :
            if outvalue < c_value :
                c_r = c_seq
                c_value = outvalue
            inlist[tonid].append(c_r)
            c_seq += 1

            
# 설정 정보 직접 정의
c_alpha = 2  # Alpha 값 직접 설정
nnc_calculation = False  # NNC 계산 여부, 필요하다면 사용자 입력을 통해 설정 가능
            
            
# nc값을 저장할 배열 초기화
nc_values = []
for inrankvalues in inlist:
    nc_value = 0.0
    if nnc_calculation:  # NNC 산출
        if inrankvalues != []:
            for inrankvalue in inrankvalues:
                if inrankvalue == 1:  # 1위인 경우의 수 count
                    nc_value += 1.0
    else:  # 1/r^Alpha 로 합산
        if inrankvalues != []:
            for inrankvalue in inrankvalues:
                nc_value += 1.0/(inrankvalue**c_alpha)
    nc_values.append(nc_value)


####################################

# 노드별 neighbor centrality를 nodelist에 추가한 후 sort
for nodenum in range(1,n_nodes+1) :
    nodelist[nodenum].append(nc_values[nodenum])

nodelist.sort()


####################################

# 노드별 neighbor centrality 출력 준비
if nnc_calculation:
    outstr = "[NODE]\t[NNC]\n"
else:
    if c_alpha == 0:
        outstr = "[NODE]\t[DC]\n"
    else:
        outstr = "[NODE]\t[NC]\n"
        


        # 출력 파일명 생성 로직 수정
nc_prefix = "zzNC_2.0"  # Alpha 값이 2로 고정되어 있는 것으로 가정
# 수정된 출력 파일명 생성 방식
outfname = os.path.join(directory, f"p{period_number}_{nc_prefix}-term_cossim_edged.txt")
        
# 수정된 출력 파일명을 사용하여 파일 쓰기
with open(outfname, 'w', encoding='utf-8') as outf:
    __t__ = outf.write(outstr)
    for nodenum in range(1, n_nodes + 1):
        __t__ = outf.write(f"{nodelist[nodenum][0]}\t{format(nodelist[nodenum][3], '.10f')}\n")

print(f"'{outfname}' is successfully generated.")