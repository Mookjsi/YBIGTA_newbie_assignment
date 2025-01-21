### Advertising.csv를 불러와 데이터 로드하기!
advertising_data <- read.csv("Advertising.csv")

### Multiple Linear Regression을 수행해봅시다!
model <- lm(sales ~ TV + radio + newspaper, data = advertising_data)
coefficients <- model_summary$coefficients
print(round(coefficients, 3))


### Correlation Matrix를 만들어 출력해주세요!
correlation_matrix <- cor(advertising_data[, c("TV", "radio", "newspaper", "sales")])
print(round(correlation_matrix, 3))

