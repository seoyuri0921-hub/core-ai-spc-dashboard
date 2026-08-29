const { chromium } = require('playwright');

const DASHBOARD_URL =
  'https://core-ai-spc-dashboard-kyfg6yxrrfjntwabxcfqqf.streamlit.app/';

(async () => {

  const browser = await chromium.launch({
    headless: false,
    slowMo: 150
  });

  const context = await browser.newContext({
    viewport: {
      width: 1600,
      height: 900
    },
    recordVideo: {
      dir: 'videos/',
      size: {
        width: 1600,
        height: 900
      }
    }
  });

  const page = await context.newPage();

  console.log('1. Dashboard 접속');

  await page.goto(DASHBOARD_URL, {
    waitUntil: 'domcontentloaded',
    timeout: 120000
  });


  // =====================================================
  // Streamlit iframe
  // =====================================================

  let appFrame = null;

  for (let i = 0; i < 100; i++) {

    appFrame = page.frames().find(
      frame => frame.url().includes('/~/+/')
    );

    if (appFrame) break;

    await page.waitForTimeout(200);
  }

  if (!appFrame) {
    console.log('❌ Streamlit App Frame 없음');
    await browser.close();
    return;
  }

  console.log('2. App Frame 연결 완료');

  await appFrame
    .getByText('AI–SPC Process Monitor')
    .waitFor({
      state: 'visible',
      timeout: 30000
    });


  // =====================================================
  // 특정 가로 블록만 화면에 맞추는 함수
  // =====================================================

  async function fitBlock(keywords, scale = 0.72) {

    const blocks =
      appFrame.locator(
        '[data-testid="stHorizontalBlock"]'
      );

    const count =
      await blocks.count();

    for (let i = 0; i < count; i++) {

      const block =
        blocks.nth(i);

      const text =
        await block
          .innerText()
          .catch(() => '');

      const matched =
        keywords.every(
          keyword => text.includes(keyword)
        );

      if (matched) {

        await block.evaluate(
          (el, scaleValue) => {

            el.style.transform =
              `scale(${scaleValue})`;

            el.style.transformOrigin =
              'top left';

            el.style.width =
              '100%';

            el.style.marginRight =
              '0';

          },
          scale
        );

        console.log(
          `✅ 화면 맞춤: ${keywords.join(' / ')}`
        );

        return true;
      }
    }

    return false;
  }


  // =====================================================
  // Lot 3
  // =====================================================

  console.log('3. Lot 3 선택');

  const lotCombo =
    appFrame.getByRole(
      'combobox',
      { name: 'Lot 선택' }
    );

  await lotCombo.click();

  await appFrame.getByRole(
    'option',
    {
      name: '3',
      exact: true
    }
  ).click();

  await appFrame.getByText(
    /Lot 3\s*\|/
  ).first().waitFor({
    state: 'visible',
    timeout: 10000
  });

  console.log('✅ Lot 3');

  await page.waitForTimeout(1300);


  // =====================================================
  // W6
  // =====================================================

  console.log('4. W6 선택');

  const waferCombo =
    appFrame.getByRole(
      'combobox',
      { name: '현재 Wafer' }
    );

  await waferCombo.click();

  await appFrame.getByRole(
    'option',
    {
      name: '6',
      exact: true
    }
  ).click();

  await appFrame.getByText(
    /Lot 3.*현재 W6.*다음 W7 예측/
  ).first().waitFor({
    state: 'visible',
    timeout: 10000
  });

  console.log('✅ Lot 3 / W6');

  await page.waitForTimeout(700);


  // =====================================================
  // STEP 1 / 2 / 3 행만 축소
  // =====================================================

  await fitBlock(
    ['STEP 1', 'STEP 2', 'STEP 3'],
    0.70
  );

  await appFrame
    .locator('[data-testid="stMain"]')
    .evaluate(el => {
      el.scrollTop = 0;
    });


  // =====================================================
  // 73%
  // =====================================================

  const risk73 =
    appFrame.getByText(
      '73%',
      { exact: true }
    ).first();

  await risk73.waitFor({
    state: 'visible',
    timeout: 10000
  });

  console.log('5. ✅ 73% 사전경고');

  await page.waitForTimeout(2800);


  // =====================================================
  // 상세분석 열기
  // =====================================================

  console.log('6. 상세 분석');

  const detail =
    appFrame.getByText(
      '🔎 상세 분석 보기',
      { exact: true }
    );

  await detail.scrollIntoViewIfNeeded();

  await page.waitForTimeout(600);

  await detail.click();

  await page.waitForTimeout(1200);


  // =====================================================
  // SPC 기준 숫자 행 맞춤
  // 현재 식각 깊이 / LCL / 중심선 / UCL
  // =====================================================

  await fitBlock(
    [
      '현재 식각 깊이',
      'LCL',
      '중심선',
      'UCL'
    ],
    0.78
  );


  // =====================================================
  // SHAP 주요 공정 신호 행 맞춤
  // =====================================================

  await fitBlock(
    [
      '#1 주요 공정 신호',
      '#2 주요 공정 신호',
      '#3 주요 공정 신호'
    ],
    0.72
  );


  // =====================================================
  // SHAP 화면으로 이동
  // =====================================================

  const shapTitle =
    appFrame.getByText(
      'AI 판단에 기여한 주요 공정 신호',
      { exact: true }
    );

  if (await shapTitle.count()) {

    await shapTitle.scrollIntoViewIfNeeded();

    console.log('7. ✅ SHAP 화면');

    await page.waitForTimeout(2600);
  }


  // =====================================================
  // 실제 W7 결과
  // =====================================================

  console.log('8. 실제 W7 결과');

  const actual =
    appFrame.getByText(
      '🧪 실제 다음 Wafer 결과 확인',
      { exact: true }
    );

  if (await actual.count()) {

    await actual.scrollIntoViewIfNeeded();

    await page.waitForTimeout(700);

    await actual.click();

    await page.waitForTimeout(2200);
  }


  // =====================================================
  // 종료
  // =====================================================

  console.log('✅ Demo 완료');

  await page.waitForTimeout(1500);

  await context.close();
  await browser.close();

})();