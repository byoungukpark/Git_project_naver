# 주제
- 스마트 컨트랙트를 이용한 게임, ui 포함 코드 입니다.  

- 실행 환경
    - Truffle v5.11.2 (core: 5.11.2)

    - Ganache v7.9.0

    - Solidity v0.5.16 (solc-js)

    - Node v18.17.1

    - Web3.js v1.10.0

- 실행 방법  
    1. crypto_wordle 파일 다운

    2. cd contracts

    3. truffle comfile

    4. truffle migrate >> 나온 컨트랙트 주소를 contracts_app/src/config.js 파일안에 export const CONTACT_ADDRESS = '';로 저장

    5. cd contracts_app

    6. yarn install

    7. yarn start

###
+ 참고 : 가나슈에서 이벤트로그를 볼려면 truffle-config.js 파일을 TRUFFLE PROJECTS에 추가하여야 합니다.

## 실행
- ui (React)  
    ![image](https://github.com/byoungukpark/Git_project_naver/assets/88645300/e5ed542e-5930-4cf4-86b8-94b39cab250d)

- 배포된 컨트랙트 (가나슈)  
  ![image](https://github.com/byoungukpark/Git_project_naver/assets/88645300/c530af7f-6346-4eb6-a421-9a5800606c40)

- 게임 시작 로그 (가나슈)  
    ![image](https://github.com/byoungukpark/Git_project_naver/assets/88645300/2f6193cb-0467-41be-b63f-16d38f600e7e)

- 게임 종료 로그 (가나슈)
    ![image](https://github.com/byoungukpark/Git_project_naver/assets/88645300/03c0d098-00ad-4d69-8773-afdb7ba340e2)





