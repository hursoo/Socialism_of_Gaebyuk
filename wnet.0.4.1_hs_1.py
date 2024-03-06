# -*- coding: utf-8 -*-
#####################################################################
#
# WNet - Weighted Network Anaysis Program

# Version 0.4.1_hs_0.1
#   - code를 utf-8로 변경 (+ open ... encoding = "utf-8")
#   - python을 3.0 버전으로 변경 (print 함수, raw_input -> input)    

# Version 0.4.1
#   - maxtrix file 끝의 빈 데이터를 제대로 처리 못하는 오류 수정
#   - 행렬읽을 때 각 행렬요소에서 줄바꿈 문자 삭제하는 부분에서 pop을 쓰지 않고 slicing으로 처리
#
# Version 0.4
#   - Mean Profile Centrality 추가 (Mean Profile Association이라고도 함)
#     Pearson Correlation Matrix를 생성한 후 계산
#   - time.sleep() 함수로 종료 후 지연 처리
#
# Version 0.3.1
#   - Python 2.7 version ( / operation, coding, print command, raw_input() )
#   - Modify message
#
# Version 0.3
#   - PFNet, PNNC, Weighted Network Centralities
#
#####################################################################

###########################
#
# Node name list : proxmat[0]에 있음 ( proxmat[0][1] ... proxmat[0][n] )



import sys
from time import sleep as waiting
from os.path import exists

print("---------------------------------------------------------")
print("WNET (Weighted Network analysis)")
print(" - PFNet, PNNC, and Weighted Network Centralities (v.0.4,1)")
print("\t\t\t\tby J. Y. Lee")
print("---------------------------------------------------------\n")

# 처리할 시기의 총 개수 또는 특정 시기 번호 입력 받기 (예제에서는 단일 시기를 예로 들었습니다)
# 이 부분은 사용자의 구체적인 요구에 맞게 조정할 수 있습니다.
period_number = int(input("처리할 시기의 번호를 입력하세요: "))


# 자동으로 파일 이름 설정
simfname = f'result/p{period_number}_term_cossim.txt'
if not exists(simfname):
    print(f"File {simfname} doesn't exist")
    waiting(1.5)
    sys.exit()                
    

# PFNet, PNNC, WCENT 파일 이름도 이전 단계에서 생성된 파일 이름에 기반하여 자동으로 설정
outfname = f'result/p{period_number}_zPFNet-term_cossim.txt'
pnncfname = f'result/p{period_number}_zPNNC-term_cossim.txt'
centfname = f'result/p{period_number}_zWCENT-term_cossim.txt'

#########################################
# Reading proximity matrix
simf = open(simfname,'r', encoding='utf-8')
lines = simf.readlines()
simf.close()

proxmat = []

# 각 줄에서 행렬요소를 분할하기
for element in lines:
	proxmat.append(element.split('\t'))

# 각 행렬요소에서 줄바꿈 문자 삭제하기 (각 줄 요소 리스트의 마지막 요소를 줄바꿈문자가 없는 것으로 바꿔치기 함)
for element in proxmat:
# modified at version 0.4.1
        element[len(element)-1] = element[len(element)-1][:-1]
#	p = element.pop()
#	element.append(p.replace('\n',''))

n_node = len(proxmat[0]) - 1
print("\n...",n_node,'X',n_node,"size matrix is entered")

# 정방행렬보다 줄 수가 더 많을 경우에 나머지 무시
if len(proxmat) > (n_node+1) :
        proxmat = proxmat[:n_node+1]
        
# Matrix validation
for x in proxmat :
        if len(x) != (n_node + 1) :
                outstr = "Input matrix is not a square matrix"
                print(outstr)
                waiting(1.5)
                sys.exit()
                
# convert string to float
for listi in proxmat[1:] :
	for j in range(1,n_node+1) :
		listi[j] = float(listi[j])


#########################################
# Network generate

pfnets = []   # PFNet을 구성하는 링크쌍 [x,y]로 채우게 됨

n_cl = n_node   # number of clusters initialize
n_links = 0

newcmembers = []        # 군집소속 item 번호를 초기화
newcnum = [] # 각 item의 새 군집번호를 초기화
pnnc_history = []
for ele in range(n_cl+1):
        newcmembers.append([ele])
        newcnum.append(ele)
        pnnc_history.append([])
#        pnnc_history.append([ele])
# pnnc_history[0] = [n_cl]

cproxmat = []   # 군집 간 연관성 행렬 초기화
for ele in proxmat :
        cproxmat.append(ele)


n_cl = n_node # 군집 수 초기화

while n_cl > 1: # 군집 수가 1개로 줄면 중단

        oldcmembers = newcmembers
        newcmembers = [[]]
        oldcnum = newcnum
        newcnum = []
        for i in range(n_node+1) :
                newcnum.append(0)
        n_newc = 0

        # 지난 단계 군집번호와 소속을 old로 복사, new는 빈 것으로 다시 초기화
        pnnc_history[0].append(n_cl)    # 0번 자리에는 현 step의 군집 수 저장
        for x, pnncnum in enumerate(oldcnum[1:]) :
                pnnc_history[x+1].append(pnncnum)

        # 군집 간 거리 산출 (cproxmat[][]을 새로 채움)
        cproxmat = [[]]
        for x in range(1,n_cl+1) :
                cproxmat.append([x])
                for y in range(1,n_cl+1) :
                        # x와 y 군집 소속 item간 최고 proximity를 구함
                        cproxmat[x].append( max( [proxmat[memberx][membery] for memberx in oldcmembers[x] for membery in oldcmembers[y] ] ) )

        # 군집 간 최근접 거리 파악
        nnsimlist = [0]
        for i in range(1,n_cl+1) :
                nnsimlist.append(sorted(cproxmat[i][1:], reverse=True)[1])

        # 군집별 최근접 이웃 nnclist[1:n_cl] 파악, 동률 최근접이면 모두 추가
        nnclist = [[]]
        for ele in range(1,n_cl+1) :
                nnclist.append([])
        for i, proxs in enumerate(cproxmat[1:]) :
                for j in range(1,n_cl+1) :
                        if (proxs[j] == nnsimlist[i+1] and (i+1) != j ) :       # nnsim이 자신과 유사도와 같아서 자신과의 링크가 생성되는 것을 방지
                                nnclist[i+1].append(j)

        # for debugging
        # print oldcmembers
        
        # 각 군집별 최근접 이웃 (nnclist[i])을 찾아서 링크 생성
        for i, nncs in enumerate(nnclist[1:]) :
                c_c = i + 1

                for nnc in nncs :       # 복수 최근접을 모두 처리
                        
                        #########################################
                        ### c_c와 nnc와의 링크 생성 ###
                        #########################################
                        # 두 군집 사이의 최근접 proximity값 파악
                        maxproxvalue = max([ proxmat[x][y] for x in oldcmembers[c_c] for y in oldcmembers[nnc] ])
                        # 최근접 proximity에 해당하는 쌍을 모두 링크 생성 [proximity, itemx, itemy]
                        maxproxlink = [ [proxmat[x][y],x,y] for x in oldcmembers[c_c] for y in oldcmembers[nnc] if (proxmat[x][y] == maxproxvalue and x != y) ]
                        # 새로 생성된 링크가 기존 링크와 중복되지 않으면 pfnets에 추가
                        # - 쌍방 링크를 방지하려면, newlink의 [1]과 [2]를 맞바꾼 것도 있는지 확인해야 함 #################
                        for newlink in maxproxlink :
                                # if not (newlink in pfnets) :
                                if not ( (newlink in pfnets) or ([newlink[0],newlink[2],newlink[1]] in pfnets) ) :
                                        pfnets.append(newlink)
                        
                        #########################################
                        ### c_c와 nnc 두 군집의 병합 ###
                        #########################################

                        newcnum_c_c = newcnum[oldcmembers[c_c][0]]      # c_c 소속 멤버의 새 군집번호를 c_c의 새 군집번호로 기억
                        newcnum_nnc = newcnum[oldcmembers[nnc][0]]      # nnc 소속 멤버의 새 군집번호를 nnc의 새 군집번호로 기억

                        # 둘 다 미편입이면, - 새 군집 생성
                        if newcnum_c_c == newcnum_nnc == 0 :
                                # 두 군집의 소속 item을 새 군집에 추가
                                newcmembers.append(oldcmembers[c_c] + oldcmembers[nnc])
                                # 기존 군집 소속 item의 새 군집번호 지정
                                n_newc += 1
                                for member in oldcmembers[c_c] :
                                        newcnum[member] = n_newc
                                for member in oldcmembers[nnc] :
                                        newcnum[member] = n_newc

                        # 군집 c_c만 미편입이면, 군집 nnc의 차기 군집에다가 c_c 소속 item을 추가 
                        elif newcnum_c_c == 0 and newcnum_nnc != 0 :
                                # c_c 소속 item을 nnc의 새 군집에 추가
                                newcmembers[newcnum_nnc] += oldcmembers[c_c]
                                # c_c 소속 itme의 새 군집번호 지정
                                for member in oldcmembers[c_c] :
                                        newcnum[member] = newcnum_nnc
                                
                        # 군집 nnc만 미편입이면, 군집 c_c의 차기 군집에다가 nnc 소속 item을 추가 
                        elif newcnum_c_c != 0 and newcnum_nnc == 0 :
                                # nnc 소속 item을 c_c의 새 군집에 추가
                                newcmembers[newcnum_c_c] += oldcmembers[nnc]
                                # nnc 소속 itme의 새 군집번호 지정
                                for member in oldcmembers[nnc] :
                                        newcnum[member] = newcnum_c_c

                        # nnc와 c_c 모두 새 군집이 정해져 있으면, 두 새 군집을 병합
                        else :
                                # c_c가 작은 군집번호이면 c_c로 nnc 소속 item을 옮김
                                if newcnum_c_c < newcnum_nnc :
                                        newcmembers[newcnum_c_c] += newcmembers[newcnum_nnc]
                                        for member in newcmembers[newcnum_nnc] :
                                                newcnum[member] = newcnum_c_c
                                        # nnc 이후 군집번호를 1씩 줄임
                                        for indexx, cnumx in enumerate(newcnum) :
                                                if cnumx > newcnum_nnc :
                                                        newcnum[indexx] -= 1
                                        # newcmembers에서 nnc를 삭제
                                        newcmembers.pop(newcnum_nnc)
                                        n_newc -= 1
                                # nnc가 작은 군집번호이면 nnc로 c_c 소속 item을 옮김
                                elif newcnum_c_c > newcnum_nnc :
                                        newcmembers[newcnum_nnc] += newcmembers[newcnum_c_c]
                                        for member in newcmembers[newcnum_c_c] :
                                                newcnum[member] = newcnum_nnc
                                        # c_c 이후 군집번호를 1씩 줄임
                                        for indexx, cnumx in enumerate(newcnum) :
                                                if cnumx > newcnum_c_c :
                                                        newcnum[indexx] -= 1
                                        # newcmembers에서 c_c를 삭제
                                        newcmembers.pop(newcnum_c_c)
                                        n_newc -= 1
                                # nnc와 c_c가 새 군집에서는 이미 같은 군집이면, 아무 일 없음 
                                else :
                                        pass

        n_cl = n_newc   # 현 단계 군집수 저장


################## PFNet 만들기 종료 (전체가 모두 연결됨)



#########################################
# Network print (link and link weights)

nodelabel = proxmat[0]

outf = open(outfname,'w', encoding='utf-8')
outstr = "[Node1]\t[Node2]\t[Weight]\n"
__t__ = outf.write(outstr)
for c_link in pfnets :
        __t__ = outf.write(nodelabel[c_link[1]] + '\t' + nodelabel[c_link[2]] + '\t' + ('%.5f' % c_link[0]) + '\n')
outf.close()
outstr = "\n'" + outfname + "' is successfully generated."
print(outstr)

#########################################
# PNNC print (item label, serial number, cluster number(s) of each step)

outf = open(pnncfname,'w', encoding='utf-8')
outstr = "[Item]\t[SN]"
for each_step in pnnc_history[0][1:] :  # number of clusters in each step
        outstr += '\t[C' + '%d' % each_step + ']'
__t__ = outf.write(outstr)


for x, each_history in enumerate(pnnc_history[1:]) :
        outstr ='\n' + nodelabel[x+1]
        for stepnum in each_history :
                outstr += '\t' + '%d' % stepnum
        __t__ = outf.write(outstr)
outf.close()

outstr = "'" + pnncfname + "' is successfully generated."
print(outstr)


#############################################################
# Weighted Centrality Calculation
#############################################################

n_possible_links = ( n_node * (n_node - 1) ) / 2.0

min_prox = float(proxmat[1][1])        # 최저값 초기화
nn_prox = [0]   # 각 node의 최근접 이웃 링크값

#### 최저값 파악
for x in range(1,n_node+1) :
        c_max = 0.0
        for y in range(1,n_node+1) :
                if (proxmat[x][y] > c_max) and (x != y) :       # 최고링크값 파악
                        c_max = proxmat[x][y]
                if (x < y ) and (proxmat[x][y] < min_prox) :
                        min_prox = proxmat[x][y]
        nn_prox.append(c_max)


#### Mean Proximities (Calculating Cm)
simavg = [0.0]
sum_ni = [0.0]  # for Cmp calculation - correlation formulae
sumsq_ni = [0.0]  # for Cmp calculation - correlation formulae
for x in range(1,n_node+1) :
        simsum = 0.0
        simsqsum = 0.0
        for y in range(1,n_node+1) :
                if x != y:
                        simsum += proxmat[x][y]
                        simsqsum += proxmat[x][y]**2
        simavg.append(simsum/float(n_node - 1))
        sum_ni.append(simsum + proxmat[x][x])           # for Cmp calculation
        sumsq_ni.append(simsqsum + + proxmat[x][x]**2 )       # for Cmp calculation


#### Triad Betweenness Centrality (Calculating Ctb)
tbc = [0.0]
for x in range(1,n_node+1) :
        tbcvalue = 0
        for y in range(1,n_node+1) :
                if y == x:
                        continue
                for z in range(y+1,n_node+1) :
                        if z == x:
                                continue
                        if (proxmat[y][z] < proxmat[x][y]) and (proxmat[y][z] < proxmat[x][z]) :
                                tbcvalue += 1
        tbc.append(tbcvalue)


#### Nearest Neighbor Centrality (Calculating Cn)

# 최저값(대부분 0)이 아니면서 최고링크값에 해당하는 상대방 노드에 최근접이웃 빈도 부여

nnc = [0]
nn_list = [[]]  # 최근접 이웃 목록 초기화
for x in range(1,n_node+1) :
        nn_list.append([])
        
for x in range(1,n_node+1) :
        c_nnc = 0
        for y in range(1,n_node+1) :
                if x == y :     # 자신은 제외
                        continue
                if (proxmat[x][y] == nn_prox[y]) and (proxmat[x][y] > min_prox ) :
                        c_nnc += 1
                        nn_list[y].append(proxmat[0][x])        # y의 최근접 이웃 목록에 x 추가
        nnc.append(c_nnc)


#### Mean Profile Centrality (Calculating Cmp, also known as Mean Profile Association)
#      - Pearson Correlation Matrix를 생성한 후 계산
#      - sum(n_i), sumsq(n_i) 미리 계산

# Correlation matrix generation
corrmat = [ [0.0] * (n_node+1) for x in range(n_node+1) ]
for i in range(1,n_node+1) :
        corrmat[i][i] = 1.0
for x in range(1,n_node) :
        for y in range(x+1,n_node+1) :
                sumproduct = 0.0
                for z in range(1,n_node+1) :
                        sumproduct += proxmat[x][z] * proxmat[y][z]
                corrmat[x][y] = corrmat[y][x] = ( n_node * sumproduct - sum_ni[x] * sum_ni[y] ) / ( ( (n_node * sumsq_ni[x] - (sum_ni[x])**2)**0.5) * ((n_node * sumsq_ni[y] - (sum_ni[y])**2)**0.5) )

# Cmp calculcation
cmp = [0.0]
for x in range(1,n_node+1) :
        cmp.append((sum(corrmat[x]) - 1.0)/float(n_node-1))

#########################################
# Centralities print (Serial number, Node label, TBC, rTBC, AVGSIM, NNC, rNNC, NN(s))

outf = open(centfname,'w', encoding='utf-8')
__t__ = outf.write("SN\tNODE\tTBC\trTBC(0~1)\tAVGSIM\tCmp(-1~1)\tNNC\trNNC(0~1)\tNNs")
 
for x in range(1,n_node+1) :
        nns_str = ""
        for element in nn_list[x] :
                nns_str += ('\t' + element)     # 최근접 이웃의 목록 문자열 생성
        output_str = '\n' + str(x) + '\t' + proxmat[0][x] + '\t' + str(tbc[x]) + '\t' + ( "%.5f" % (tbc[x]/float((n_node-1)*(n_node-2)/2.0)) )
        output_str = output_str + '\t' + ( "%.5f" % simavg[x] ) + '\t' + ( "%.5f" % cmp[x] ) + '\t' + str(nnc[x]) + '\t' + ( "%.5f" % (nnc[x]/float(n_node - 1)) )
        output_str = output_str + nns_str
        __t__ = outf.write(output_str)
outf.close()

outstr = "'" + centfname + "' is successfully generated."
print(outstr)
waiting(2)
# notice_str = "\n\tPress enter key..."
# __t__ = raw_input(notice_str)

