/*

fft based whisperization
Algorithm is based on dafx - U. Zoelzer page 290 ff
Project is deployed on 
 ____  _____ _        _    
| __ )| ____| |      / \   
|  _ \|  _| | |     / _ \  
| |_) | |___| |___ / ___ \ 
|____/|_____|_____/_/   \_\

The platform for ultra-low latency audio and sensor processing

http://bela.io

The Bela software is distributed under the GNU Lesser General Public License
(LGPL 3.0), available here: https://www.gnu.org/licenses/lgpl-3.0.txt
*/

#include <Bela.h>
#include <libraries/ne10/NE10.h>
//#include <libraries/Midi/Midi.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#define BUFFER_SIZE (16384)

// set the frequency of the oscillators
float gInputBuffer[BUFFER_SIZE];
int gInputBufferPointer = 0;
float gOutputBuffer[BUFFER_SIZE];
float inL, inR, outL, outR;
int gOutputBufferWritePointer = 0;
int gOutputBufferReadPointer = 0;
int gSampleCount = 0;

// These variables used internally in the example:
int gFFTSize = 512;
// phase parameters
int Ha = gFFTSize/8-0; /* analysis hopsize */
int Hs = gFFTSize/8;   /* synthisis hopsize */
int gHopSize = Hs;
int gPeriod = gHopSize;
float modWhisp = 0.f; // 0...1, where 0 means no whisperization and 1 full whisperization
float prevModWhisp = 0.f; 

float gFFTScaleFactor = 0;
float *gInputAudio = NULL;
float *gWindowBuffer;

void process_whisperization_background(void *);

// FFT vars
ne10_fft_cpx_float32_t* timeDomainIn;
ne10_fft_cpx_float32_t* timeDomainOut;
ne10_fft_cpx_float32_t* frequencyDomain;
ne10_fft_cfg_float32_t cfg;

// phase processing vars
float *phi;
float *amplitude;

// Set the analog channels to read from
const int gAnalogIn = 0;
int gAudioFramesPerAnalogFrame = 0;

int gReadPtr = 0;        // Position of last read sample from file
AuxiliaryTask gFFTTask;
int gFFTInputBufferPointer = 0;
int gFFTOutputBufferPointer = 0;

// cpu cycle read
static inline uint32_t ccnt_read (void)
{
  uint32_t cc = 0;
  __asm__ volatile ("mrc p15, 0, %0, c9, c13, 0":"=r" (cc));
  return cc;
}

// instantiate the scope
//Scope scope;

// userData holds an opaque pointer to a data structure that was passed
// in from the call to initAudio().
//
// Return true on success; returning false halts the program.
bool setup(BelaContext* context, void* userData)
{
    printf("go setup\n");
    printf("context->audioFrames = %d\n", context->audioFrames);
    printf("context->audioSampleRate = %f\n", context->audioSampleRate);
    printf("context->audioInChannels = %d\n", context->audioInChannels);
    printf("context->audioOutChannels = %d\n", context->audioOutChannels);
    // Check that we have the same number of inputs and outputs.
    if(context->audioInChannels != context->audioOutChannels ||
            context->analogInChannels != context-> analogOutChannels){
        printf("Error: for this project, you need the same number of input and output channels.\n");
        return false;
    }

    gFFTScaleFactor = (float)Hs / (float)gFFTSize;
    gOutputBufferWritePointer = Hs;
    gOutputBufferReadPointer = 0;

    timeDomainIn = (ne10_fft_cpx_float32_t*) NE10_MALLOC (gFFTSize * sizeof (ne10_fft_cpx_float32_t));
    timeDomainOut = (ne10_fft_cpx_float32_t*) NE10_MALLOC (gFFTSize * sizeof (ne10_fft_cpx_float32_t));
    frequencyDomain = (ne10_fft_cpx_float32_t*) NE10_MALLOC (gFFTSize * sizeof (ne10_fft_cpx_float32_t));
    cfg = ne10_fft_alloc_c2c_float32_neon (gFFTSize);
    
    memset(timeDomainIn, 0, gFFTSize * sizeof (ne10_fft_cpx_float32_t));
    memset(timeDomainOut, 0, gFFTSize * sizeof (ne10_fft_cpx_float32_t));
    memset(gOutputBuffer, 0, BUFFER_SIZE * sizeof(float));
    memset(gInputBuffer, 0, BUFFER_SIZE * sizeof(float));

    // Allocate phase processing buffer and init vars
    phi = (float *)malloc(gFFTSize * sizeof(float));
    if(phi == 0)
        return false;
    amplitude = (float *)malloc(gFFTSize * sizeof(float));
    if(amplitude == 0)
        return false;
        
    // Allocate buffer to mirror and modify the input
    gInputAudio = (float *)malloc(context->audioFrames * context->audioOutChannels * sizeof(float));
    if(gInputAudio == 0)
        return false;

    // Allocate the window buffer based on the FFT size
    gWindowBuffer = (float *)malloc(gFFTSize * sizeof(float));
    if(gWindowBuffer == 0)
        return false;

    // Calculate a Hann window
    for(int n = 0; n < gFFTSize; n++) {
        gWindowBuffer[n] = 0.5f * (1.0f - cosf(2.0f * M_PI * (float)n / (float)(gFFTSize)));
    }

    // Initialise auxiliary tasks
    if((gFFTTask = Bela_createAuxiliaryTask(&process_whisperization_background, 90, "fft-calculation")) == 0)
        return false;

    // Check if analog channels are enabled
    if(context->analogFrames == 0 || context->analogFrames > context->audioFrames) {
        rt_printf("Error: this example needs analog enabled, with 4 or 8 channels\n");
        return false;
    }
    // Useful calculations
    if(context->analogFrames)
        gAudioFramesPerAnalogFrame = context->audioFrames / context->analogFrames;
        
    srand(time(NULL));   // Initialization for random

    printf("bye setup\n");
    return true;
}

// This function handles the FFT based whisperization
void process_whisperization(float *inBuffer, int inWritePointer, float *outBuffer, int outWritePointer)
{

    uint32_t t0 = ccnt_read();
    uint32_t t1 = t0;//ccnt_read();       
    //rt_printf("%u\n", t1-t0);

    // Copy buffer into FFT input
    int pointer = (inWritePointer - gFFTSize + BUFFER_SIZE) % BUFFER_SIZE;
    for(int n = 0; n < gFFTSize; n++) {
        timeDomainIn[n].r = (ne10_float32_t) inBuffer[pointer] * gWindowBuffer[n];
        timeDomainIn[n].i = 0.0f;

        pointer++;
        if(pointer >= BUFFER_SIZE)
            pointer = 0;
    }

    // Run the FFT
    ne10_fft_c2c_1d_float32_neon (frequencyDomain, timeDomainIn, cfg, 0);

    for(int n = 0; n < gFFTSize; n++) {
        amplitude[n] = sqrtf(frequencyDomain[n].r * frequencyDomain[n].r + frequencyDomain[n].i * frequencyDomain[n].i);
        phi[n] = atan2f((float)frequencyDomain[n].i, (float)frequencyDomain[n].r);
        float randVal = (float)rand()/(float)RAND_MAX;
        //printf("randVal = %f\n",randVal);
        phi[n] = modWhisp*(2.f*M_PI*randVal) + (1.f-modWhisp)*phi[n];

    }
    
    for(int n = 0; n < gFFTSize; n++) {
        frequencyDomain[n].r = cosf(phi[n]) /** (1+modWhisp)*/ * amplitude[n];
        frequencyDomain[n].i = sinf(phi[n]) /** (1+modWhisp)*/ * amplitude[n];
    }

    ne10_fft_c2c_1d_float32_neon (timeDomainOut, frequencyDomain, cfg, 1);
    
    for(int n = 0; n < gFFTSize; n++)
    {
        timeDomainOut[n].r = gFFTScaleFactor * timeDomainOut[n].r * gWindowBuffer[n];
    }

    // Overlap-and-add timeDomainOut into the output buffer
    pointer = outWritePointer;
    int n;
    for(n = 0; n < gFFTSize; n++)
    {
        outBuffer[pointer] += timeDomainOut[n].r;
        
        //if(isnan(outBuffer[pointer]))
        //    rt_printf("outBuffer OLA\n");

        pointer++;
        if(pointer >= BUFFER_SIZE)
            pointer = 0;
    }

    t1 = ccnt_read();
    rt_printf("\rmodWhisp = %f ####  %u cycles process", modWhisp, t1-t0);

}

// Function to process the FFT in a thread at lower priority
void process_whisperization_background(void*) {
    process_whisperization(gInputBuffer, gFFTInputBufferPointer, gOutputBuffer, gFFTOutputBufferPointer);
}

void render(BelaContext *context, void *userData)
{
    // iterate over the audio frames and create three oscillators, seperated in phase by PI/2
    for(unsigned int n = 0; n < context->audioFrames; n++) {
        if(gAudioFramesPerAnalogFrame && !(n % gAudioFramesPerAnalogFrame)) {
            if(n==0)
            {
                // read analog inputs and update modWhisp value
                modWhisp = (float)floor(map(analogRead(context, n/gAudioFramesPerAnalogFrame, gAnalogIn), 0, 1, 0, 110))/100.f;
#if 1
                if(modWhisp >= prevModWhisp+0.1f)
                {
                    prevModWhisp += 0.1f;
                }
                else if(modWhisp <= prevModWhisp-0.1f)
                {
                    prevModWhisp -= 0.1f;
                }

                modWhisp = prevModWhisp;
#endif

            }
        }

        // Read audio inputs
        inL = audioRead(context,n,0);
        inR = audioRead(context,n,1);
#if 1 // apply whisperization if defined. otherwise it's bypassed
        gInputBuffer[gInputBufferPointer] = (inR+inL) * 0.5f;
            
        outL = gOutputBuffer[gOutputBufferReadPointer];
        outR = outL;
        
        // Clear the output sample in the buffer so it is ready for the next overlap-add
        gOutputBuffer[gOutputBufferReadPointer] = 0;
        
        gOutputBufferReadPointer++;
        if(gOutputBufferReadPointer >= (BUFFER_SIZE))
            gOutputBufferReadPointer = 0;

        gOutputBufferWritePointer++;
        if(gOutputBufferWritePointer >= (BUFFER_SIZE))
            gOutputBufferWritePointer = 0;

        gInputBufferPointer++;
        if(gInputBufferPointer >= (BUFFER_SIZE))
            gInputBufferPointer = 0;

        gSampleCount++;
        if(gSampleCount >= Hs) {
#if 0
            /* do not use scheduling */
            process_whisperization(gInputBuffer, gInputBufferPointer, gOutputBuffer, gOutputBufferWritePointer);
#else
            gFFTInputBufferPointer = gInputBufferPointer;
            gFFTOutputBufferPointer = gOutputBufferWritePointer;
            Bela_scheduleAuxiliaryTask(gFFTTask);
#endif
            gSampleCount = 0;
        }    
        
#else
        outL = inL;
        outR = inR;
#endif        
        audioWrite(context, n, 0, outL);
        audioWrite(context, n, 1, outR);
    }
}

// cleanup_render() is called once at the end, after the audio has stopped.
// Release any resources that were allocated in initialise_render().

void cleanup(BelaContext* context, void* userData)
{
    NE10_FREE(timeDomainIn);
    NE10_FREE(timeDomainOut);
    NE10_FREE(frequencyDomain);
    NE10_FREE(cfg);

    free(gWindowBuffer);
    free(phi);
    free(amplitude);
}
