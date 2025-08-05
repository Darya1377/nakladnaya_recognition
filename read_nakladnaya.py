import cv2
from ultralytics import YOLO
import numpy as np
import re

def parse_string(input_str):
    # Ищем номер (может быть одно- или двузначным)
    number_match = re.match(r'^(\d{1,2})', input_str)
    if number_match:
        number = number_match.group(1)
        remaining_str = input_str[len(number):].strip()
    
    # Разделяем оставшуюся часть на компоненты
    # Ищем наименование (до количества)
    name_match = re.match(r'^([^\d]+)', remaining_str)
    if not name_match:
        return "Не удалось найти наименование"
    
    name = name_match.group(1).strip()
    remaining_str = remaining_str[len(name_match.group(1)):].strip()
    
    # Ищем количество (целое число)
    quantity_match = re.match(r'^(\d+)', remaining_str)
    if not quantity_match:
        return "Не удалось найти количество"
    
    quantity = quantity_match.group(1)
    remaining_str = remaining_str[len(quantity):].strip()
    
    # Ищем единицу измерения (буквы до цифр цены)
    unit_match = re.match(r'^([^\d]+)', remaining_str)
    if not unit_match:
        return "Не удалось найти единицу измерения"
    
    unit = unit_match.group(1).strip()
    remaining_str = remaining_str[len(unit_match.group(1)):].strip()
    
    # Проверяем, что осталось достаточно символов для цены и суммы
    if len(remaining_str) < 5:
        return "Недостаточно данных для цены и суммы"
    
    # Последние 4 символа - сумма, остальное - цена
    price = remaining_str[:-4].strip()
    total = remaining_str[-4:].strip()
    
    # Формируем результат
    result = f"номер: {number}, наименование: {name}, количество: {quantity}, единица измерения: {unit}, цена: {price}, сумма: {total}"
    return result

def detect_and_print_letters(image_path, model_path=r'C:\Users\user\Desktop\img2text\runs\detect\train16\weights\best.pt'):

    model = YOLO(model_path)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    results = model(image)

    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        class_names = result.names
        
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            detections.append({
                'class': class_names[int(cls_id)],
                'confidence': float(conf),
                'box': box,
                'center': (center_x, center_y),
                'width': width,
                'height': height
            })
    
    if not detections:
        print("No letters detected in the image.")
        return

    detections.sort(key=lambda x: x['center'][1])
    
    rows = []
    current_row = [detections[0]]
    avg_height = detections[0]['height']

    for det in detections[1:]:
        if abs(det['center'][1] - current_row[0]['center'][1]) < avg_height * 0.6:
            current_row.append(det)
            avg_height = sum(d['height'] for d in current_row) / len(current_row)
        else:
            current_row.sort(key=lambda x: x['center'][0])
            rows.append(current_row)
            current_row = [det]
            avg_height = det['height']

    if current_row:
        current_row.sort(key=lambda x: x['center'][0])
        rows.append(current_row)

    print("\nDetected text (top to bottom, left to right with spacing):\n")

    header_processed = 0
    shift_fields = False
    left_out = 0

    for i, row in enumerate(rows):
        if header_processed >= 4:  # We've processed all 4 header rows
            break
            
        row_text = ""
        prev_right = -np.inf
        
        for det in row:
            x1 = det['box'][0]
            if prev_right > -np.inf:
                gap = x1 - prev_right
                space_count = max(0, int(round(gap / det['width'])))
                if space_count > 0.5:
                    row_text += ' ' * space_count
            row_text += det['class']
            prev_right = det['box'][2]
        
        if i == 0:
            date_parts = re.findall(r'\d+', row_text)
            if len(date_parts) >= 3:
                print(f'Дата: {date_parts[0]}.{date_parts[1]}.20{date_parts[2]}')
            else:
                print(f'Дата: {row_text}')
            header_processed += 1
        elif i == 1:
            if row_text.isdigit():
                print(f'Накладная № {row_text}')
            else:
                print('Отсутствует номер накладной')
                left_out = 1
                shift_fields = True
                print('Кому ', row_text)
            header_processed += 1
        elif i == 2:
            if not shift_fields:
                print('Кому ', row_text)
            else:
                print('От кого ', row_text)
            header_processed += 1
        elif i == 3 and not shift_fields:
            print('От кого ', row_text)
            header_processed += 1
    print("\nItems:")
    items = []
    i = 4 - left_out 

    while i < len(rows):
        row_text = "".join(det['class'] for det in rows[i]).strip()

        if i == len(rows) - 1:
            if row_text.isdigit(): 
                print(f"\nИтого: {row_text}")
                break
            else:
                print("Итого нет")
        if re.match(r'^\d', row_text):

            ordinal = re.match(r'^\d+', row_text).group()
            remaining_text = row_text[len(ordinal):].strip()
            item_match = re.match(r'^([^\d]+)(\d+)\s*([^\d]+)?\s*(\d{1,2})\s*(\d+)$', remaining_text)
            if item_match:
                name = item_match.group(1).strip()
                quantity = item_match.group(2)
                unit = item_match.group(3).strip() if item_match.group(3) else None
                price = item_match.group(4)
                item_sum = item_match.group(5)
                
                items.append({
                    'ordinal': ordinal,
                    'name': name,
                    'quantity': quantity,
                    'unit': unit,
                    'price': price,
                    'sum': item_sum
                })
        i += 1

    for item in items:
        output = f"{item['ordinal']}. {item['name']} {item['quantity']}"
        if item['unit']:
            output += f" {item['unit']}"
        if item['price'] and item['sum']:
            output += f" | {item['price']} {item['sum']}"
        print(output)

    for row in rows:
        for det in row:
            x1, y1, x2, y2 = map(int, det['box'])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, det['class'], (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow("Detected Letters", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = r"C:\Users\user\Desktop\img2text\new_test\1108716_800.jpg"  # Replace with your image path
    detect_and_print_letters(image_path)

