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

## 실행 테스트
