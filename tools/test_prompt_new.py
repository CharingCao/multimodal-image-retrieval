from extended.analyze_full import analyze_full

images = [
    "data/image_input/01.png",
    "data/image_input/02.png",
    "data/image_input/03.png",
    "data/image_input/04.png",
    "data/image_input/05.png",
    "data/image_input/06.png",
    "data/image_input/07.png",
    "data/image_input/08.png",
    "data/image_input/09.png",
    "data/image_input/10.png",
    "data/image_input/11.png",
    "data/image_input/12.png",
    "data/image_input/13.png",
    "data/image_input/14.png",
    "data/image_input/15.png",
    "data/image_input/16.png",
    "data/image_input/17.png"

]


for img in images:
    print("="*50)
    print(img)

    result = analyze_full(img)

    print(result)