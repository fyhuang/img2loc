import Jimp from "jimp/es";

export async function loadImageFromPath(path: string, size = 224) {
    var imageData = Jimp.read(path).then((image) => {
        return image.resize(size, size);
    }).catch((err: any) => {
        console.error(err);
    });

    return imageData;
}

async function userFileToBase64(file: File): Promise<string> {
    const fileReader = new FileReader();
    return new Promise((resolve, reject) => {
        fileReader.addEventListener("load", () => {
            resolve(fileReader.result as string);
        });
        fileReader.addEventListener("error", () => {
            reject(fileReader.error);
        });
        fileReader.readAsDataURL(file);
    });
}

export class ImageInferenceHandler {
    previewImgEl: HTMLImageElement;
    canvasEl: HTMLCanvasElement;

    constructor(previewImgEl, canvasEl) {
        this.previewImgEl = previewImgEl;
        this.canvasEl = canvasEl;

        previewImgEl.addEventListener("load", (e) => {
            var ctx = this.canvasEl.getContext("2d");
            ctx?.drawImage(e.target, 0, 0, this.canvasEl.width, this.canvasEl.height);
            const data = ctx?.getImageData(0, 0, this.canvasEl.width, this.canvasEl.height);
            this.processData(data as ImageData);
        });
    }

    async processUserFile(file: File) {
        const imgBase64 = await userFileToBase64(file);
        this.previewImgEl.src = imgBase64;
    }

    processData(data: ImageData) {
        console.log(data.data[0]);
    }
}