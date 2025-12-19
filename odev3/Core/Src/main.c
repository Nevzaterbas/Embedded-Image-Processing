/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body for EE4065 HW3 (Clean Version)
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include <string.h>
#include <math.h>
#include <stdint.h>

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define IMG_WIDTH  64
#define IMG_HEIGHT 64
#define IMG_SIZE   (IMG_WIDTH * IMG_HEIGHT)
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
uint8_t image_buffer[IMG_SIZE];
uint8_t temp_buffer[IMG_SIZE];
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
/* USER CODE BEGIN PFP */
void Apply_Otsu(uint8_t* img, int size);
void Apply_Erosion(uint8_t* src, uint8_t* dest);
void Apply_Dilation(uint8_t* src, uint8_t* dest);
uint8_t get_pixel(uint8_t* img, int x, int y);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

// --- YARDIMCI: Piksel Okuma ---
uint8_t get_pixel(uint8_t* img, int x, int y) {
    if (x < 0 || x >= IMG_WIDTH || y < 0 || y >= IMG_HEIGHT) return 0;
    return img[y * IMG_WIDTH + x];
}

// --- OTSU (Q1) ---
void Apply_Otsu(uint8_t* img, int size) {
    int histogram[256] = {0};
    for (int i = 0; i < size; i++) histogram[img[i]]++;

    float sum_total = 0;
    for (int i = 0; i < 256; i++) sum_total += i * histogram[i];

    float sum_bg = 0;
    int w_bg = 0, w_fg = 0;
    float var_max = 0;
    int threshold = 0;

    for (int t = 0; t < 256; t++) {
        w_bg += histogram[t];
        if (w_bg == 0) continue;
        w_fg = size - w_bg;
        if (w_fg == 0) break;

        sum_bg += (float)(t * histogram[t]);
        float m_bg = sum_bg / w_bg;
        float m_fg = (sum_total - sum_bg) / w_fg;

        float var_bet = (float)w_bg * (float)w_fg * (m_bg - m_fg) * (m_bg - m_fg);

        if (var_bet > var_max) {
            var_max = var_bet;
            threshold = t;
        }
    }

    for (int i = 0; i < size; i++) {
        img[i] = (img[i] > threshold) ? 255 : 0;
    }
}

// --- EROSION (Q3) ---
void Apply_Erosion(uint8_t* src, uint8_t* dest) {
    for (int y = 0; y < IMG_HEIGHT; y++) {
        for (int x = 0; x < IMG_WIDTH; x++) {
            uint8_t min_val = 255;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    uint8_t val = get_pixel(src, x + kx, y + ky);
                    if (val < min_val) min_val = val;
                }
            }
            dest[y * IMG_WIDTH + x] = min_val;
        }
    }
    memcpy(src, dest, IMG_SIZE);
}

// --- DILATION (Q3) ---
void Apply_Dilation(uint8_t* src, uint8_t* dest) {
    for (int y = 0; y < IMG_HEIGHT; y++) {
        for (int x = 0; x < IMG_WIDTH; x++) {
            uint8_t max_val = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    uint8_t val = get_pixel(src, x + kx, y + ky);
                    if (val > max_val) max_val = val;
                }
            }
            dest[y * IMG_WIDTH + x] = max_val;
        }
    }
    memcpy(src, dest, IMG_SIZE);
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
  MX_USART2_UART_Init();

  /* USER CODE BEGIN WHILE */
  while (1)
  {
    // PC'den 64x64 resim bekle (Timeout: 5000ms)
    if (HAL_UART_Receive(&huart2, image_buffer, IMG_SIZE, 5000) == HAL_OK)
    {
        // 1. ADIM: Otsu
        Apply_Otsu(image_buffer, IMG_SIZE);

        // 2. ADIM: Morfolojik (Q3) - Closing ornegi
        Apply_Dilation(image_buffer, temp_buffer);
        Apply_Erosion(image_buffer, temp_buffer);

        // 3. ADIM: Geri Yolla
        HAL_UART_Transmit(&huart2, image_buffer, IMG_SIZE, 1000);
    }
    HAL_Delay(50);
  }
  /* USER CODE END WHILE */
}

/**
  * @brief System Clock Configuration
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 16;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief USART2 Initialization Function
  */
static void MX_USART2_UART_Init(void)
{
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief GPIO Initialization Function
  */
static void MX_GPIO_Init(void)
{
  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  // LED ve Buton kodlari temizlendi (hata vermemesi icin)
}

void Error_Handler(void)
{
  __disable_irq();
  while (1)
  {
  }
}

#ifdef  USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
}
#endif
