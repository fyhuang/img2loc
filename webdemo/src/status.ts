export class StatusDisplay {
    private divElement: HTMLDivElement;

    constructor(divId: string) {
        this.divElement = document.getElementById(divId) as HTMLDivElement;
    }

    log(message: string) {
        this.divElement.innerText = message;
        this.divElement.classList.remove('error');
    }

    error(error: string) {
        this.divElement.innerText = error;
        this.divElement.classList.add('error');
    }
}