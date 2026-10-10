from search_weight import search_similar_packaging
import os

TEST_IMAGES=[]

for i in range(1,51):

    for ext in [
        "png",
        "jpg",
        "jpeg",
        "PNG"
    ]:

        path=f"data/image_input/{i:02d}.{ext}"

        if os.path.exists(path):
            TEST_IMAGES.append(path)
            break
        
GROUND_TRUTH = {
    "01.png":"4cc914fda48bcfc89f637ec237990fa4.png",
    "02.png":"402619870b28731554d0af16a1978d72.png",
    "03.png":"7f7135989c064b2d68701be6ad67e13c.png",
    "04.png":"504551bfc52ecb77982d05dd2c346911.png",
    "05.png":"c478e4064cc705ea90557042fdbba550.png",
    "06.png":"692b4cd2a9d08d0ff9707e42db47c07e.png",
    "07.png":"21a74c411fbbf86984ec13199778954e.png",
    "08.png":"2adc2f4f0cc6b0de7a5731e0ce60c193.png",
    "09.png":"ba32ec1c15565092b109c5b3c1915085.png",
    "10.png":"65fdba5e922a7e0d56cc23ffdc81b4fd.png",
    "11.png":"14ffb60db14121ee9e798931d7716cb9.png",
    "12.png":"45e9364b28b6effea73688a6216b2f2c.png",
    "13.png":"734eb8cb8ad654d8f7ebba473868390f.png",
    "14.png":"dbd87e688924470121079183088bbb7d.png",
    "15.png":"5838139afe732671298b8fd1c70ef503.png",
    "16.png":"507fba32d397bab0fdcf18c19880c5cb.png",
    "17.png":"832a09050d36ec8ad3c0fd4cf8b2b5d9.png",
    "18.png":"cf71cf29314d796611c2dc06b52d5a96.png",
    "19.png":"d74e58b106d2d331247e0075b759af05.png",
    "20.png":"d7085156e22a7745c4a1be46279c5a3f.png",
    "21.png":"7f9658a1723107aa0fac32993b907d48.png",
    "22.png":"5a878189bc6f6c616edd8758a8d5947f.png",
    "23.png":"15cb8aa21e3e91c00024569faeb7e382.png",
    "24.png":"31aa1a3da11ad68873b97df17774a74d.png",
    "25.png":"cd5f4a03fa8741c27e24c3bf542aacbc.png",
    "26.png":"787c6d58ddac90cf6ee1c757e77fb049.png",
    "27.png":"52528bdc793fb82c561551104a029417.png",
    "28.png":"b42061398125b59f1bf0e791add5faa4.png",
    "29.png":"4767f154be5158f627da44260b8ec340.png",
    "30.png":"300c633b3cb55545134bddef270b9614.png"
}

WEIGHTS=[]

for clip in [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65
]:

    for tier1 in [
        0.05,
        0.10,
        0.15
    ]:

        for tier2 in [
            0.20,
            0.25,
            0.30,
            0.35
        ]:

            tier3=1-clip-tier1-tier2

            if tier3>=0:

                WEIGHTS.append(
                    {
                        "clip":clip,
                        "tier1":tier1,
                        "tier2":tier2,
                        "tier3":round(tier3,2)
                    }
                )

print(
    "权重数量:",
    len(WEIGHTS)
)

def evaluate(weight):

    top1=0
    top10=0

    for img in TEST_IMAGES:

        results, desc = search_similar_packaging(img, weight, 10)

        names=[
            x["filename"]
            for x in results
        ]

        target=GROUND_TRUTH[
            img.split("/")[-1]
        ]

        if names:

            if names[0]==target:
                top1+=1

            if target in names:
                top10+=1

    n=len(TEST_IMAGES)

    top1_rate=top1/n
    top10_rate=top10/n

    score=(
        top1_rate*0.8+
        top10_rate*0.2
    )

    return (
        top1_rate,
        top10_rate,
        score
    )

records=[]

for i,w in enumerate(WEIGHTS):

    print(
        f"\n{i+1}/{len(WEIGHTS)}",
        w
    )

    t1,t10,s=evaluate(w)
    records.append(
        {
            "score":s,
            "top1":t1,
            "top10":t10,
            "weight":w
        }
    )

    print(
        f"Top1:{t1:.2%}",
        f"Top10:{t10:.2%}",
        f"Score:{s:.4f}"
    )

records.sort(
    key=lambda x:x["score"],
    reverse=True
)

print(
    "\n最佳10组"
)

for r in records[:10]:

    print(
        "\n",
        r
    )