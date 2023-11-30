import torch
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# question =  "戒壇院四天王像の国宝指定名称は何?"
# context = "戒壇院四天王像は国宝だ。奈良時代。塑造。像高持国天160.5センチ、増長天162.2センチ、広目天169.9センチ、多聞天164.5センチである。国宝指定名称は「塑造四天王立像」だ。天平勝宝7年(755年)、唐僧鑑真によって設立された東大寺戒壇院の中心堂宇である戒壇堂に安置される。戒壇堂内の壇上中央に多宝塔があり、これを護るように壇上四隅に四天王像が立つ。現存する戒壇堂と多宝塔は享保18年(1733年)の再建であるが、四天王像は奈良時代の作である。ただし、戒壇堂に本来安置されていた四天王像は銅造であったことが史料からわかっており、現在安置されている四天王像(塑造)は材質が異なるため、後世他所から移入された像であることが明らかである。壇上、東南隅に東方を守護する持国天像が立ち、以下、西南隅に南方守護の増長天像、西北隅に西方守護の広目天像、東北隅に北方守護の多聞天像が立つ。"
# # answer_label = "「塑造四天王立像」"

# context = "松江騒擾事件（まつえそうじょうじけん）は、1945年（昭和20年）8月24日未明、日本の島根県松江市で青年グループ「皇国義勇軍」数十人が武装蜂起し、県内主要施設を襲撃したという事件である。この事件により、民間人1名が死亡した。"
# question = "松江騒擾事件はどこで発生しましたか？"

# context = "名古屋工業大学（なごや こうぎょうだいがく、英語: Nagoya Institute of Technology、公用語表記: 名古屋工業大学）は、愛知県名古屋市昭和区御器所町に本部を置く日本の国立大学。1905年創立、1949年大学設置。大学の略称は名工（めいこう）、名工大（めいこうだい）。"
# question = "名古屋工業大学はどこですか？"

# # その中で、ベトナムで最も裕福な男であるPham Nhat Vuong氏の富は約22億米ドル減少した。今年の初め以来、VingroupのVIC株は半分近く減少し、12月26日に取引セッションを1ユニットあたり約53,000VNDで終了しました。 現在、Vuong氏は40億米ドル相当の資産しか所有していません。2017年以来の最低レベルです。
# # ベトナムで最も裕福な人は誰ですか？
# context = "Trong đó, khối tài sản của ông Phạm Nhật Vượng, người giàu nhất Việt Nam, giảm khoảng 2,2 tỷ USD. Từ đầu năm đến nay, cổ phiếu VIC của Vingroup đã giảm gần một nửa, chốt phiên giao dịch 26/12 ở mức gần 53.000 đồng mỗi đơn vị.Hiện tại, ông Vượng chỉ sở hữu khối tài sản trị giá 4 tỷ USD – mức thấp nhất kể từ năm 2017."
# question = "Ai là người giàu nhất Việt Nam?"

with open("./test_data.json", encoding="utf-8") as f:
    test_data = json.loads(f.read())["test_data"]

for dt in test_data:
    context = dt["context"]
    question = dt["question"]

    inputs = tokenizer(question, context, return_tensors="pt")

    model = torch.load("./fine_tuned_model_epoch_3.pth", map_location="cpu")
    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()
    # print(answer_start_index, answer_end_index)

    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    answer = tokenizer.decode(predict_answer_tokens)

    print(f"Context: {context}\n")
    print(f"Question: {question}\n")
    print(f"Answer: {answer}\n")
    print("-"*100, "\n")
